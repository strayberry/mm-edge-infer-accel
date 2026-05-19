from mm_edge_infer_accel.datasets import stratified_select


class FakeDataset:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def select(self, indices):
        return [self.items[index] for index in indices]


def test_stratified_select_round_robins_sorted_groups():
    ds = FakeDataset(
        [
            {"question_type": "b", "id": 0},
            {"question_type": "a", "id": 1},
            {"question_type": "b", "id": 2},
            {"question_type": "a", "id": 3},
            {"question_type": "c", "id": 4},
        ]
    )

    selected = stratified_select(ds, 4, key="question_type")

    assert [item["id"] for item in selected] == [1, 0, 4, 3]
