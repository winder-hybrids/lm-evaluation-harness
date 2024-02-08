"""
Generated multiple choice QA tasks.
Structure of this file is modeled after the MMLU benchmark (hendrycks_test.py)
"""

from lm_eval.base import MultipleChoiceTask

names = ['Anatomy_Gray', 'Biochemistry_Lippincott', 'Cell_Biology_Alberts', 'Gynecology_Novak', 'Histology_Ross', 'Immunology_Janeway', 'Neurology_Adams', 'Obstentrics_Williams', 'Pathology_Robbins', 'Pediatrics_Nelson', 'Pharmacology_Katzung', 'Physiology_Levy', 'Psichiatry_DSM-5']


def create_all_tasks():
    return {
        name: create_task("winder-hybrids/MedicalTextbook_QA", name) for name in names
    }

def create_task(dataset_name, subset):
    class MC(GeneralMC):
        def __init__(self):
            super().__init__(dataset_name, subset)

    return MC


class GeneralMC(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, dataset_name, subset):
        self.DATASET_PATH = dataset_name
        self.DATASET_NAME = subset
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["test"])
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
            
        def format_example(doc):
            """
            <prompt> 
            """

            question = doc["question"].strip()
            prompt = f"{question}"
            return prompt

        return {
            "query": format_example(doc),
            "choices": doc["choices"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
