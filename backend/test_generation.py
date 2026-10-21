"""Deterministic checks for quotation provenance and bounded retries."""
import json
import unittest
from unittest.mock import Mock
from generation import generate, validate_evidence, sentence_choices, resolve_selection

CONTEXT = [{'rank': 1, 'passage_id': 7, 'text': 'I keep my bread in plastic. It has not molded for me.'}]


def evidence(quote, source=1):
    return {'evidence': [{'source': source, 'quote': quote}]}


class GenerationChecks(unittest.TestCase):
    def run_responses(self, *payloads):
        post = Mock(side_effect=[Mock(json=Mock(return_value=p)) for p in payloads])
        result = generate('How do I store homemade bread?', CONTEXT, post, 'http://local', 'model')
        return result, post

    def test_exact_personal_quote_preserves_scope(self):
        data = evidence('I keep my bread in plastic.')
        result, post = self.run_responses({'response': json.dumps(data), 'done_reason': 'stop'})
        self.assertEqual(result['evidence_quotes'], data['evidence'])
        self.assertNotIn('homemade', result['answer'])
        self.assertEqual(post.call_count, 1)

    def test_fabrication_and_dropped_negation_rejected(self):
        for quote in ['I keep my homemade bread in plastic.', 'It has molded for me.']:
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                validate_evidence(evidence(quote), CONTEXT)
        restored = validate_evidence(evidence('molded for me.'), CONTEXT)
        self.assertEqual(restored[0]['quote'], 'It has not molded for me.')

    def test_invalid_citation_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            validate_evidence(evidence('I keep my bread in plastic.', 9), CONTEXT)
        with self.assertRaises(ValueError):
            validate_evidence({'evidence': evidence('I keep my bread in plastic.')['evidence'] * 2}, CONTEXT)

    def test_truncated_output_replaced_by_completed_retry(self):
        result, post = self.run_responses(
            {'response': '{', 'done_reason': 'length'},
            {'response': json.dumps({'sentence_ids': [1]}), 'done_reason': 'stop'})
        self.assertEqual(result['answer_status'], 'quoted_evidence')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result['generation_attempts'][0]['problem'], 'output_limit')

    def test_repeated_bad_output_is_not_shown(self):
        bad = {'response': json.dumps(evidence('All bread stays fresh forever.'))}
        result, post = self.run_responses(bad, bad)
        self.assertEqual(result['answer_status'], 'generation_failed')
        self.assertEqual(result['citations'], [])
        self.assertNotIn('forever', result['answer'])
        self.assertEqual(post.call_count, 2)

    def test_selection_uses_exact_text_and_preserves_source_rank(self):
        contexts = [{'rank': 9, 'text': 'Do not water daily. Water only when dry.'}]
        choices = sentence_choices(contexts)
        result = resolve_selection({'sentence_ids': [1]}, choices, contexts)
        self.assertEqual(result, [{'source': 9, 'quote': 'Do not water daily.'}])
        for ids in [[99], [True], [1, 1], [1, 2]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                resolve_selection({'sentence_ids': ids}, choices, contexts)
        self.assertEqual(resolve_selection({'sentence_ids': []}, choices, contexts), [])

    def test_abstention_and_reasoning_handling(self):
        result, post = self.run_responses({'response': '{"evidence": []}'})
        self.assertEqual(result['answer_status'], 'insufficient_evidence')
        self.assertFalse(result['warnings'])
        bad = {'response': '<think>hidden reasoning</think>'}
        result, _ = self.run_responses(bad, bad)
        self.assertNotIn('hidden reasoning', result['answer'])


if __name__ == '__main__':
    unittest.main()

