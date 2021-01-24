from torch_gauge.o3.clebsch_gordan import get_clebsch_gordan_coefficient


def test_generate_cg():
    max_j = 4
    for j1 in range(max_j + 1):
        for j2 in range(max_j + 1):
            for j in range(abs(j1 - j2), max_j + 1):
                print(f"Generating Clebsch-Gordan coefficients for j1={j1}, j2={j2}, j={j}")
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m1 + m2)
