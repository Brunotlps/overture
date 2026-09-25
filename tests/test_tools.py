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

    def test_skips_symlinks_and_case_variants_of_sensitive_paths(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "safe.txt").write_text("public")
        (repo / "TOKEN.txt").write_text("private")
        (repo / "Secrets").mkdir()
        (repo / "Secrets" / "data.txt").write_text("private")
        (repo / "safe-link.txt").symlink_to("safe.txt")
        (repo / "outside-link.txt").symlink_to(tmp_path / "outside.txt")
        (repo / "broken.txt").symlink_to("missing.txt")
        (repo / "loop.txt").symlink_to("loop.txt")

        assert list_files(str(repo)) == ["safe.txt"]


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

    def test_rejects_directory_with_clear_error(self, fake_repo):
        with pytest.raises(ValueError, match="is not a file"):
            read_file(str(fake_repo), "src")

    def test_rejects_symlink_alias_to_sensitive_file(self, tmp_path):
        (tmp_path / "API.KEY").write_text("marker")
        (tmp_path / "alias.txt").symlink_to("API.KEY")

        with pytest.raises(ValueError):
            read_file(str(tmp_path), "alias.txt")

    def test_rejects_symlink_directory_and_sensitive_parent(self, tmp_path):
        (tmp_path / "TOKEN_store").mkdir()
        (tmp_path / "TOKEN_store" / "data.txt").write_text("marker")
        (tmp_path / "alias").symlink_to("TOKEN_store", target_is_directory=True)

        with pytest.raises(ValueError):
            read_file(str(tmp_path), "alias/data.txt")
        with pytest.raises(ValueError, match="sensitive data"):
            read_file(str(tmp_path), "TOKEN_store/data.txt")


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

    def test_does_not_follow_symlinks_outside_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (tmp_path / "outside.txt").write_text("private-marker")
        (repo / "leak.txt").symlink_to(tmp_path / "outside.txt")

        assert grep_repo(str(repo), "private-marker") == []
