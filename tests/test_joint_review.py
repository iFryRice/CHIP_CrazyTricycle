import copy
import unittest
from patientphex.joint_review import changed_examples


class JointReviewTests(unittest.TestCase):
    def setUp(self):
        self.row = {"task_id":"a", "pmc_id":"1", "patient_id":"p1", "concept":"HP:0001250",
                    "input_ids":[1,2,3], "attention_mask":[1,1,1], "token_type_ids":[0,0,1], "occurrences":3}

    def test_reuses_identical_tokens_but_recomputes_changed_context(self):
        unchanged = {**self.row, "occurrences":2}
        self.assertEqual(changed_examples([self.row], [unchanged]), [])
        changed = {**unchanged, "input_ids":[1,4,3]}
        self.assertEqual(changed_examples([self.row], [changed]), [changed])
        self.assertEqual(changed_examples([self.row], []), [])

    def test_rejects_new_or_reassigned_candidates(self):
        for key,value in [("task_id","new"),("patient_id","p2")]:
            altered = {**self.row,key:value}
            with self.assertRaises(ValueError): changed_examples([self.row],[altered])
        with self.assertRaises(ValueError): changed_examples([self.row],[self.row,copy.deepcopy(self.row)])


if __name__ == "__main__": unittest.main()
