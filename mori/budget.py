"""Conservative prompt accounting without a model-specific tokenizer or network access."""

COUNTING_METHOD = 'utf8_bytes'
COUNTING_NOTE = (
    'Conservative estimate: one token per UTF-8 byte, not an exact model token count. '
    'This can leave unused context. Output and adapter overhead are reserved separately; '
    'the provider may add context beyond Mori\'s prompt.'
)


class ContextBudget:
    """Snapshot the configured allowance and count the complete final answer prompt.

    UTF-8 bytes deliberately avoid an assumed characters-per-token ratio. They
    bound text tokens for byte-level tokenizers, but are not a guarantee about an
    arbitrary provider's tokenizer, hidden instructions or generated output.
    """

    def __init__(self, config):
        config.validate()
        self.limit_tokens = config.context_window_tokens
        self.output_reserve_tokens = config.llm_max_output_tokens
        self.overhead_reserve_tokens = config.context_overhead_tokens
        self.input_budget_tokens = (self.limit_tokens - self.output_reserve_tokens
                                    - self.overhead_reserve_tokens)
        self.input_char_limit = config.llm_max_input_chars

    @staticmethod
    def count(prompt):
        if not isinstance(prompt, str):
            raise TypeError('The context budget counts a complete prompt string')
        return len(prompt.encode('utf-8'))

    def measure(self, prompt):
        prompt_tokens = self.count(prompt)
        return {
            'mode': 'context',
            'limit_tokens': self.limit_tokens,
            'output_reserve_tokens': self.output_reserve_tokens,
            'overhead_reserve_tokens': self.overhead_reserve_tokens,
            'input_budget_tokens': self.input_budget_tokens,
            'prompt_tokens': prompt_tokens,
            'remaining_tokens': self.input_budget_tokens - prompt_tokens,
            'counting_method': COUNTING_METHOD,
            'counting_note': COUNTING_NOTE,
        }

    def fits(self, prompt):
        return (self.count(prompt) <= self.input_budget_tokens
                and len(prompt) <= self.input_char_limit)
