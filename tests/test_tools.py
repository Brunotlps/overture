import tracemalloc

import pytest

from app.tools import (
    MAX_FILE_LINES,
    MAX_GREP_LINE_CHARS,
    MAX_LINE_CHARS,
    MAX_READ_CHARS,
    grep_repo,
    list_files,
    read_file,
)


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

    def test_numbers_lines_and_points_to_continuation(self, fake_repo):
        lines = read_file(str(fake_repo), "src/big_file.py").splitlines()

        assert lines[0] == "1: line 0"
        assert lines[MAX_FILE_LINES - 1] == f"{MAX_FILE_LINES}: line {MAX_FILE_LINES - 1}"
        assert lines[-1] == (
            "... [more lines follow; call read_file with start_line=301 to continue]"
        )
        assert len(lines) == MAX_FILE_LINES + 1

    def test_continuation_reads_the_rest_of_the_file(self, fake_repo):
        lines = read_file(str(fake_repo), "src/big_file.py", start_line=301).splitlines()

        assert lines[0] == "301: line 300"
        assert lines[-1] == "500: line 499"
        assert len(lines) == 200

    def test_definition_after_line_300_is_found_by_grep_and_readable(self, tmp_path):
        lines = [f"filler_{i} = {i}" for i in range(1, 451)]
        lines[399] = "def late_definition():"
        lines[400] = "    return 'context after the cut'"
        (tmp_path / "late.py").write_text("\n".join(lines) + "\n")

        [match] = grep_repo(str(tmp_path), "late_definition")
        line_number = int(match.split(":")[1])
        content = read_file(
            str(tmp_path), "late.py", start_line=line_number - 2, max_lines=5
        )

        assert match == "late.py:400: def late_definition():"
        assert content.splitlines() == [
            "398: filler_398 = 398",
            "399: filler_399 = 399",
            "400: def late_definition():",
            "401:     return 'context after the cut'",
            "402: filler_402 = 402",
            "... [more lines follow; call read_file with start_line=403 to continue]",
        ]

    def test_last_window_has_no_continuation(self, fake_repo):
        content = read_file(str(fake_repo), "src/big_file.py", start_line=499)

        assert content.splitlines() == ["499: line 498", "500: line 499"]

    @pytest.mark.parametrize(
        ("start_line", "max_lines", "message"),
        [
            (0, 10, "start_line must be 1 or greater"),
            (-5, 10, "start_line must be 1 or greater"),
            (1, 0, "max_lines must be between 1 and 300"),
            (1, MAX_FILE_LINES + 1, "max_lines must be between 1 and 300"),
            (501, 10, "start_line 501 is past the end of 'src/big_file.py' \\(500 lines\\)"),
        ],
    )
    def test_rejects_invalid_ranges(self, fake_repo, start_line, max_lines, message):
        with pytest.raises(ValueError, match=message):
            read_file(str(fake_repo), "src/big_file.py", start_line, max_lines)

    def test_empty_file_reads_as_empty_but_has_no_later_lines(self, tmp_path):
        (tmp_path / "empty.py").write_text("")

        assert read_file(str(tmp_path), "empty.py") == ""
        with pytest.raises(ValueError, match="past the end"):
            read_file(str(tmp_path), "empty.py", start_line=2)

    def test_ranges_keep_the_file_security_policy(self, fake_repo):
        with pytest.raises(ValueError, match="sensitive data"):
            read_file(str(fake_repo), ".env", start_line=1, max_lines=1)
        with pytest.raises(ValueError, match="outside repository"):
            read_file(str(fake_repo), "../../../etc/passwd", start_line=2)
        with pytest.raises(ValueError, match="binary"):
            read_file(str(fake_repo), "gateway", start_line=1, max_lines=1)

    def test_caps_a_very_long_line(self, fake_repo):
        content = read_file(str(fake_repo), "src/minified.json")
        omitted = len('{"minified_payload": "' + "x" * 5000 + '"}') - MAX_LINE_CHARS

        assert content.startswith('1: {"minified_payload": "xxx')
        assert content.endswith(f" ... [line truncated: {omitted} more characters]")
        assert len(content) < MAX_LINE_CHARS + 100

    def test_caps_total_output_and_continues_from_the_first_omitted_line(
        self, tmp_path
    ):
        (tmp_path / "wide.txt").write_text(("y" * 1500 + "\n") * 50)

        lines = read_file(str(tmp_path), "wide.txt").splitlines()
        shown = len(lines) - 1

        assert len("\n".join(lines[:-1])) <= MAX_READ_CHARS
        assert shown < 50
        assert lines[-1] == (
            f"... [more lines follow; call read_file with start_line={shown + 1}"
            " to continue]"
        )

    def test_reads_with_bounded_memory(self, tmp_path):
        (tmp_path / "one_line.js").write_text("z" * 10_000_000)
        (tmp_path / "many_lines.log").write_text(
            "".join(f"row {i}\n" for i in range(200_000))
        )

        tracemalloc.start()
        try:
            read_file(str(tmp_path), "one_line.js")
            read_file(str(tmp_path), "many_lines.log", max_lines=10)
            read_file(str(tmp_path), "many_lines.log", start_line=199_990)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 1_000_000

    def test_line_numbers_match_grep_for_all_newline_styles(self, tmp_path):
        (tmp_path / "mixed.txt").write_bytes(b"alpha\r\nbeta\rgamma\ndelta")

        assert grep_repo(str(tmp_path), "gamma") == ["mixed.txt:3: gamma"]
        assert read_file(str(tmp_path), "mixed.txt", start_line=3).splitlines() == [
            "3: gamma",
            "4: delta",
        ]

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

    def test_searches_long_lines_only_up_to_the_shared_line_cap(self, tmp_path):
        (tmp_path / "bundle.js").write_text(
            "early_marker" + "x" * MAX_LINE_CHARS + "late_marker\n"
        )

        assert grep_repo(str(tmp_path), "early_marker") == [
            "bundle.js:1: early_marker" + "x" * (MAX_GREP_LINE_CHARS - 12) + "... [truncated]"
        ]
        assert grep_repo(str(tmp_path), "late_marker") == []

    def test_scans_files_with_bounded_memory(self, tmp_path):
        (tmp_path / "one_line.js").write_text("z" * 10_000_000)

        tracemalloc.start()
        try:
            assert grep_repo(str(tmp_path), "needle") == []
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < 1_000_000

    def test_does_not_follow_symlinks_outside_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (tmp_path / "outside.txt").write_text("private-marker")
        (repo / "leak.txt").symlink_to(tmp_path / "outside.txt")

        assert grep_repo(str(repo), "private-marker") == []
