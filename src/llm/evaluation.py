from langsmith import Client
from langchain.smith import RunEvalConfig
from langchain.evaluation import EvaluatorType

from config import LANGCHAIN_API_KEY


class LLMEvaluation:
    def __init__(self):
        self.client = Client(api_key=LANGCHAIN_API_KEY)
    
    def evaluate_responses(self, dataset_name, chain, project_name="rag-weather-agent"):
        """
        Evaluate LLM responses using LangSmith
        """
        eval_config = RunEvalConfig(
            evaluators=[
                EvaluatorType.QA,
                EvaluatorType.CRITERIA,
                {"name": "criteria", "criteria": "correctness"}
            ],
            project_name=project_name
        )
        
        evaluation_results = self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain=chain,
            evaluation=eval_config
        )
        
        return evaluation_results