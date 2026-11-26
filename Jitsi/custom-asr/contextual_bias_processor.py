from typing import Dict, List, Tuple, Optional
import torch
from transformers import LogitsProcessor


class InterviewContextLogitsProcessor(LogitsProcessor):
    """
    Contextual terminology biasing using a token trie.

    CHANGE 2 + CB-2:
    - Do NOT bias trie root candidates on every decoding step.
    - Bias only when the generated suffix is already a plausible
      prefix of one or more contextual terminology targets.
    - Once a prefix becomes active, bias only the valid next tokens.
    - If no contextual prefix is active, leave logits unchanged.
    - CB-2 uses rank-aware adaptive biasing:
        * rank 1       -> base bias
        * rank 2-5     -> base bias + 0.5
        * rank 6-20    -> base bias + 1.0
        * rank >20     -> base bias + 1.5

    The adaptive bias is applied only to valid trie continuations.
    Normal Moonshine decoding remains untouched when no contextual
    terminology prefix is active.
    """

    def __init__(
        self,
        trie_data: Dict,
        tokenized_terms: List[Dict],
        strong_bias: float = 2.0,
        moderate_bias: float = 1.0,
        max_prefix_length: int = 16,
    ):
        super().__init__()

        self.trie = trie_data["trie"]

        self.strong_bias = float(strong_bias)
        self.moderate_bias = float(moderate_bias)
        self.max_prefix_length = int(max_prefix_length)

        self.tokenized_terms = tokenized_terms

        # token_id -> metadata
        self.token_metadata: Dict[int, List[Dict]] = {}

        self._build_token_metadata()

        self.bias_events = []

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_token_metadata(self):
        """
        Build reverse index:

            token_id -> [
                {
                    "term": ...,
                    "strength": ...,
                    "bias": ...
                },
                ...
            ]
        """

        self.token_metadata.clear()

        for term in self.tokenized_terms:
            if not term.get("valid", False):
                continue

            if not term.get("decoder_target", True):
                continue

            token_ids = term.get("token_ids", [])

            if not token_ids:
                continue

            text = term.get("term", "")

            strength = term.get(
                "decoder_strength",
                term.get("strength", "moderate"),
            )

            if strength == "strong":
                bias = self.strong_bias
            elif strength == "moderate":
                bias = self.moderate_bias
            else:
                # Weak / unknown terms are not used for decoder biasing.
                continue

            metadata = {
                "term": text,
                "strength": strength,
                "bias": bias,
                "token_ids": token_ids,
            }

            for token_id in token_ids:
                token_id = int(token_id)

                self.token_metadata.setdefault(
                    token_id,
                    [],
                ).append(metadata)

    # ------------------------------------------------------------------
    # Trie traversal
    # ------------------------------------------------------------------

    def _get_node(self, token_sequence: Tuple[int, ...]) -> Optional[Dict]:
        """
        Return trie node for an exact token sequence.
        """

        node = self.trie

        for token_id in token_sequence:
            children = node.get("children", {})

            node = children.get(str(int(token_id)))

            if node is None:
                return None

        return node

    def _find_active_prefix(
        self,
        generated_tokens: List[int],
    ) -> Tuple[int, ...]:
        """
        Find the longest suffix of generated_tokens that is a valid
        non-root trie prefix.

        IMPORTANT:
        Empty prefix is NOT considered active.

        Example:

            generated:
                [..., 7783]

            trie:
                Microsoft Sentinel
                first token = 7783

            result:
                (7783,)

        But if there is no terminology evidence:

            generated:
                [..., 512, 1034]

            result:
                ()
        """

        if not generated_tokens:
            return ()

        max_len = min(
            self.max_prefix_length,
            len(generated_tokens),
        )

        # Longest suffix first.
        for length in range(max_len, 0, -1):
            suffix = tuple(
                int(x)
                for x in generated_tokens[-length:]
            )

            node = self._get_node(suffix)

            if node is None:
                continue

            # A valid active prefix must have a child.
            #
            # If this node is terminal but has no children, the term
            # is already complete and there is nothing to bias.
            children = node.get("children", {})

            if children:
                return suffix

        return ()

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def _get_candidate_tokens(
        self,
        prefix: Tuple[int, ...],
    ) -> List[int]:
        """
        Return valid next-token IDs for the active prefix.
        """

        node = self._get_node(prefix)

        if node is None:
            return []

        children = node.get("children", {})

        return [
            int(token_id)
            for token_id in children.keys()
        ]

    def _get_candidate_terms(
        self,
        prefix: Tuple[int, ...],
    ) -> List[int]:
        """
        Return token IDs that correspond to the candidate next tokens.

        Kept as a separate method because diagnostics currently use it.
        """

        return self._get_candidate_tokens(prefix)

    # ------------------------------------------------------------------
    # Bias calculation
    # ------------------------------------------------------------------

    def _candidate_bias(
        self,
        token_id: int,
    ) -> float:
        """
        Determine the strongest bias associated with a token.

        If a token belongs to multiple contextual terms, use the
        strongest applicable bias.
        """

        metadata = self.token_metadata.get(
            int(token_id),
            [],
        )

        if not metadata:
            return 0.0

        return max(
            float(item["bias"])
            for item in metadata
        )

    def _adaptive_bias(
        self,
        base_bias: float,
        target_rank: int,
    ) -> float:
        """
        CB-2 rank-aware adaptive bias.

        Increase the bias only when the contextual target is losing
        against other vocabulary tokens.

        Rank is 1-based.

        Mapping:

            rank 1       -> base bias
            rank 2-5     -> base bias + 0.5
            rank 6-20    -> base bias + 1.0
            rank >20     -> base bias + 1.5
        """

        if target_rank <= 1:
            return base_bias

        if target_rank <= 5:
            return base_bias + 0.5

        if target_rank <= 20:
            return base_bias + 1.0

        return base_bias + 1.5

    def _candidate_term_names(
        self,
        token_id: int,
    ) -> List[str]:
        """
        Human-readable terminology names for diagnostics.
        """

        metadata = self.token_metadata.get(
            int(token_id),
            [],
        )

        return [
            item["term"]
            for item in metadata
        ]

    # ------------------------------------------------------------------
    # Main processor
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        generated = input_ids[0].tolist()

        # --------------------------------------------------------------
        # CHANGE 2:
        #
        # Never activate contextual biasing from the trie root.
        #
        # Root means:
        #   "we have no terminology evidence yet."
        #
        # Therefore:
        #   normal Moonshine decoding.
        # --------------------------------------------------------------

        prefix = self._find_active_prefix(generated)

        if not prefix:
            return scores

        node = self._get_node(prefix)

        if node is None:
            return scores

        candidate_tokens = self._get_candidate_tokens(prefix)

        if not candidate_tokens:
            return scores

        # --------------------------------------------------------------
        # Apply bias only to valid continuations of the active prefix.
        #
        # CB-2:
        # Bias magnitude is determined by the candidate's current
        # rank before the bias is applied.
        # --------------------------------------------------------------

        modified_scores = scores

        event = {
            "step": len(generated),
            "prefix": list(prefix),
            "candidate_tokens": [],
        }

        for token_id in candidate_tokens:

            base_bias = self._candidate_bias(token_id)

            if base_bias <= 0.0:
                continue

            # ----------------------------------------------------------
            # Determine the candidate's rank BEFORE applying bias.
            #
            # Rank is 1-based:
            #
            #   rank 1  = highest current logit
            #   rank 2  = second highest
            #   ...
            # ----------------------------------------------------------

            target_score = scores[0, token_id]

            target_rank = int(
                (scores[0] > target_score).sum().item()
            ) + 1

            # ----------------------------------------------------------
            # CB-2 adaptive bias.
            # ----------------------------------------------------------

            adaptive_bias = self._adaptive_bias(
                base_bias,
                target_rank,
            )

            modified_scores[:, token_id] += adaptive_bias

            event["candidate_tokens"].append(
                {
                    "token_id": int(token_id),
                    "base_bias": float(base_bias),
                    "adaptive_bias": float(adaptive_bias),
                    "target_rank_before": int(target_rank),
                    "terms": self._candidate_term_names(token_id),
                }
            )

        if event["candidate_tokens"]:
            self.bias_events.append(event)

        return modified_scores