from cases import HEAD_DIMENSIONS, MANDATORY_CASES


def test_benchmark_matrix_meets_course_coverage():
    assert len(MANDATORY_CASES) >= 6
    assert {case.D for case in MANDATORY_CASES} >= set(HEAD_DIMENSIONS)
    assert min(case.N for case in MANDATORY_CASES) < 128
    assert max(case.N for case in MANDATORY_CASES) >= 1024
    assert any(case.H > 1 for case in MANDATORY_CASES)
    assert any(case.B > 1 for case in MANDATORY_CASES)


def test_benchmark_cases_are_structurally_valid():
    names = [case.name for case in MANDATORY_CASES]
    assert len(names) == len(set(names))

    for case in MANDATORY_CASES:
        assert case.B >= 1
        assert case.H >= 1
        assert case.N >= 1
        assert case.D in HEAD_DIMENSIONS
        assert case.Dv == case.D
