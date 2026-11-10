import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import project_paths
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini_polish as gemini


class GeminiTest(unittest.TestCase):
    def setUp(self):
        self.saved = (gemini.KEY, gemini.MODEL, gemini.LAST_REQUEST)
        gemini.KEY = 'test-key-not-real'
        gemini.LAST_REQUEST = -float('inf')
        self.value = {'raw': '나/저 이름 ㅂ ㅏ ㄱ ㅎ ㅕ ㄴ ㅇ ㅜ',
                      'assembled': '나/저 이름 박현우', 'style': 'polite', 'consent': True}

    def tearDown(self):
        gemini.KEY, gemini.MODEL, gemini.LAST_REQUEST = self.saved

    def test_text_only_request_and_no_key_in_status(self):
        response = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': json.dumps({'sentence': '제 이름은 박현우입니다.'})}]}}]}
        with patch.object(gemini, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as fetch:
            self.assertEqual(gemini.polish(self.value)['sentence'], '제 이름은 박현우입니다.')
        request = fetch.call_args.args[0]
        self.assertNotIn(gemini.KEY, request.full_url)
        self.assertNotIn(gemini.KEY, request.data.decode())
        self.assertNotIn(gemini.KEY, json.dumps(gemini.status()))
        body = json.loads(request.data)
        self.assertEqual(json.loads(body['contents'][0]['parts'][0]['text']),
                         {key: self.value[key] for key in ('raw', 'assembled', 'style')})

    def test_requires_consent_and_valid_size(self):
        with patch.object(gemini, 'urlopen') as fetch:
            for value in [{**self.value, 'consent': False}, {**self.value, 'raw': 'a'*2001},
                          {**self.value, 'assembled': None}, {**self.value, 'style': 'invalid'}]:
                with self.assertRaises(ValueError):
                    gemini.polish(value)
            fetch.assert_not_called()

    def test_missing_key_and_rate_limit(self):
        gemini.KEY = ''
        with self.assertRaises(RuntimeError):
            gemini.polish(self.value)
        gemini.KEY = 'test-key-not-real'
        gemini.LAST_REQUEST = gemini.time.monotonic()
        with self.assertRaises(RuntimeError):
            gemini.polish(self.value)

    def test_upstream_secret_never_echoed(self):
        with patch.object(gemini, 'urlopen', side_effect=HTTPError('url', 403, gemini.KEY, {}, None)):
            with self.assertRaises(RuntimeError) as error:
                gemini.polish(self.value)
        self.assertNotIn(gemini.KEY, str(error.exception))

    def test_blocked_or_invalid_response(self):
        with patch.object(gemini, 'urlopen', return_value=io.BytesIO(b'{"candidates":[]}')):
            with self.assertRaises(RuntimeError):
                gemini.polish(self.value)

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            gemini.configure({'apiKey': 'test-key-not-real', 'model': '../other'})
        with self.assertRaises(ValueError):
            gemini.configure({'apiKey': 'test-key\nnot-real'})
        self.assertEqual(gemini.configure({'apiKey': 'test-key-not-real'})['configured'], True)

    def test_model_change_reuses_key_without_returning_it(self):
        result = gemini.configure({'model': 'gemini-2.5-flash-lite'})
        self.assertEqual(gemini.KEY, 'test-key-not-real')
        self.assertEqual(result['model'], 'gemini-2.5-flash-lite')
        self.assertNotIn(gemini.KEY, json.dumps(result))

    def test_low_thinking_request_uses_selected_model_and_keeps_active_model(self):
        for model, config in [('gemini-3.5-flash-lite', {'thinkingLevel':'minimal'}),
                              ('gemini-2.5-flash-lite', {'thinkingBudget':0})]:
            gemini.LAST_REQUEST = -float('inf')
            response={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"sentence":"감사합니다."}'}]}}]}
            active=gemini.MODEL
            with patch.object(gemini,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as fetch:
                result=gemini.polish(self.value,model_override=model,timeout=10)
            self.assertIn(model,fetch.call_args.args[0].full_url)
            payload=json.loads(fetch.call_args.args[0].data)
            self.assertEqual(payload['generationConfig']['thinkingConfig'],config)
            self.assertEqual(result['model'],model)
            self.assertEqual(gemini.MODEL,active)

    def test_diagnosis_uses_fixed_text_and_records_failed_and_successful_models(self):
        gemini.MODEL='gemini-3.5-flash-lite'
        def fake(value,*,model_override,timeout):
            self.assertEqual(value['raw'],'오늘 이렇게 모으다 감사합니다')
            self.assertEqual(timeout,10)
            if model_override==gemini.MODEL:
                raise RuntimeError('timeout')
            return {'model':model_override,'sentence':'오늘 이렇게 모여 주셔서 감사합니다.'}
        with patch.object(gemini,'polish',side_effect=fake),patch.object(gemini.time,'sleep'):
            result=gemini.diagnose({'consent':True})
        self.assertEqual(len(result['results']),2)
        self.assertIn('error',result['results'][0])
        self.assertIn('sentence',result['results'][1])
        self.assertNotIn(gemini.KEY,json.dumps(result))
        with self.assertRaises(ValueError):gemini.diagnose({'consent':False})

    def test_connection_errors_are_distinct_and_do_not_expose_upstream_details(self):
        cases = [(TimeoutError(gemini.KEY), '20초'),
                 (URLError(TimeoutError(gemini.KEY)), '연결 대기'),
                 (URLError(gemini.KEY), '네트워크'),
                 (HTTPError('url', 502, gemini.KEY, {}, None), 'HTTP 502')]
        for error, expected in cases:
            gemini.LAST_REQUEST = -float('inf')
            with patch.object(gemini, 'urlopen', side_effect=error), patch.object(gemini.time, 'sleep'):
                with self.assertRaises(RuntimeError) as result:
                    gemini.polish(self.value)
            self.assertIn(expected, str(result.exception))
            self.assertNotIn(gemini.KEY, str(result.exception))
            self.assertFalse(gemini.LOCK.locked())



if __name__ == '__main__':
    unittest.main()
