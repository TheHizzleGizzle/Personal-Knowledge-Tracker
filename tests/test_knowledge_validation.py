import json
import unittest

from main import app


class KnowledgeValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_content_type(self):
        resp = self.client.post(
            "/api/knowledge",
            data=json.dumps({
                "title": "bad ct",
                "content": "x",
                "content_type": "invalid",
                "tags": ["ok"],
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertFalse(body.get("success", True))
        self.assertIn("content_type", (body.get("error") or ""))

    def test_too_many_tags(self):
        tags = [f"t{i}" for i in range(1, 22)]
        resp = self.client.post(
            "/api/knowledge",
            data=json.dumps({
                "title": "too many",
                "content": "x",
                "content_type": "text",
                "tags": tags,
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertFalse(body.get("success", True))
        self.assertIn("maximum", (body.get("error") or "").lower())


if __name__ == "__main__":
    unittest.main()

