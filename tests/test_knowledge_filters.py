import json
import unittest

from main import app


class KnowledgeFilterTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _create(self, title, content_type, tags=None):
        payload = {"title": title, "content": "x", "content_type": content_type}
        if tags is not None:
            payload["tags"] = tags
        r = self.client.post(
            "/api/knowledge",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 201)
        return r.get_json()["data"]["id"]

    def test_type_filter(self):
        # Create a few items of different types
        self._create("t1", "text", ["a"])  # text
        self._create("l1", "link", ["a"])  # link
        self._create("m1", "markdown", ["b"])  # markdown

        # Filter by type=text
        r = self.client.get("/api/knowledge?type=text&per_page=50")
        self.assertEqual(r.status_code, 200)
        items = r.get_json().get("data", [])
        self.assertTrue(all(it.get("content_type") == "text" for it in items))

    def test_tag_filter(self):
        # Ensure at least one item with tag 'filtertag'
        self._create("tagged", "text", ["filtertag"])  # text with tag
        r = self.client.get("/api/knowledge?tag=filtertag&per_page=50")
        self.assertEqual(r.status_code, 200)
        items = r.get_json().get("data", [])
        # Items should include only those with that tag (server exact match)
        self.assertTrue(all(any((t.get("name") if isinstance(t, dict) else t) == "filtertag" for t in (it.get("tags") or [])) for it in items))


if __name__ == "__main__":
    unittest.main()

