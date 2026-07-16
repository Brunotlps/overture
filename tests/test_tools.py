import pytest

from app.tools import MAX_GREP_LINE_CHARS, grep_repo, list_files, read_file


class TestListFiles:
    def test_lists_files_in_repo(self, fake_repo):
        result = list_files(str(fake_repo))
        assert "README.md" in result
        assert "src/main.py" in result
        assert "src/utils.py" in result

    def test_ignores_git_directory(self, fake_repo):
        result = list_files(str(fake_repo))
        assert ".git" not in result

    def test_ignores_claude_directory(self, fake_repo):
        result = list_files(str(fake_repo))
        assert not any(path.startswith(".claude") for path in result)

    def test_filters_sensitive_files(self, fake_repo):
        result = list_files(str(fake_repo))
        assert ".env" not in result
        assert "src/api.key" not in result

    def test_raises_on_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            list_files("/path/that/does/not/exist")


class TestReadFile:
    def test_reads_small_file_fully(self, fake_repo):
        content = read_file(str(fake_repo), "README.md")
        assert "Fake Repo" in content
        assert "Projeto de teste" in content

    def test_truncates_large_file(self, fake_repo):
        content = read_file(str(fake_repo), "src/big_file.py")
        assert "line 0" in content
        assert "truncated" in content.lower()
        assert "line 499" not in content

    def test_rejects_path_traversal(self, fake_repo):
        with pytest.raises(ValueError, match="outside repository"):
            read_file(str(fake_repo), "../../../etc/passwd")

    def test_rejects_absolute_path_escape(self, fake_repo):
        with pytest.raises(ValueError, match="outside repository"):
            read_file(str(fake_repo), "/etc/passwd")

    def test_rejects_sensitive_file(self, fake_repo):
        with pytest.raises(ValueError, match="sensitive data"):
            read_file(str(fake_repo), ".env")

    def test_raises_on_missing_file(self, fake_repo):
        with pytest.raises(FileNotFoundError):
            read_file(str(fake_repo), "does_not_exist.py")

    def test_rejects_file_in_ignored_directory(self, fake_repo):
        with pytest.raises(ValueError, match="ignored directory"):
            read_file(str(fake_repo), ".claude/notes.md")

    def test_rejects_binary_file(self, fake_repo):
        with pytest.raises(ValueError, match="binary"):
            read_file(str(fake_repo), "gateway")


class TestGrepRepo:
    def test_finds_matching_term(self, fake_repo):
        results = grep_repo(str(fake_repo), "circuit_breaker")
        assert len(results) == 1
        assert "src/main.py" in results[0]

    def test_returns_empty_when_no_match(self, fake_repo):
        results = grep_repo(str(fake_repo), "termo_inexistente_xyz")
        assert results == []

    def test_limits_number_of_matches(self, fake_repo):
        results = grep_repo(str(fake_repo), "def", max_results=1)
        assert len(results) <= 1

    def test_skips_binary_files(self, fake_repo):
        results = grep_repo(str(fake_repo), "circuit_breaker")
        assert len(results) == 1
        assert "gateway" not in results[0]

    def test_skips_ignored_directories(self, fake_repo):
        results = grep_repo(str(fake_repo), "circuit_breaker")
        assert all(".claude" not in match for match in results)

    def test_truncates_long_matching_lines(self, fake_repo):
        results = grep_repo(str(fake_repo), "minified_payload")
        assert len(results) == 1
        assert results[0].endswith("... [truncated]")
        prefix = "src/minified.json:1: "
        snippet = results[0][len(prefix) : -len("... [truncated]")]
        assert len(snippet) == MAX_GREP_LINE_CHARS
