import os

from .evaluator import Evaluator
from typing import List
from pydantic import BaseModel
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from models import TechCompany


class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {
        "accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""
    }

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-turbo",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        true_answer: str = None,
        question_asked: str = None,
    ):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError(
                "true_answer and question_asked must be supplied with init."
            )

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv("NIAH_EVALUATOR_API_KEY")
        if not api_key:
            raise ValueError(
                "NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator."
            )

        self.api_key = api_key

        self.evaluator = ChatOpenAI(
            model=self.model_name, openai_api_key=self.api_key, **self.model_kwargs
        )

    def evaluate_response(self, response: List[BaseModel]) -> float:

        total_score = 0
        valid_evaluations = 0
        for company in response:
            try:
                company_score = self._evaluate_single_company(company, self.true_answer)
                total_score += company_score
                valid_evaluations += 1
            except ValueError as e:
                print(f"Error evaluating company: {e}")

        return total_score / valid_evaluations if valid_evaluations > 0 else 0

    def _evaluate_single_company(self, company: TechCompany, true_answer: str) -> int:

        print(f"Type of company in _evaluate_single_company: {type(company)}")
        if isinstance(company, TechCompany):
            company_dict = company.dict()
        elif isinstance(company, dict):
            company_dict = company
        else:
            # print(f"Unexpected company type: {type(company)}")
            company_dict = {"value": str(company)}
        company_str = ", ".join(
            [f"{k}: {v}" for k, v in company_dict.items() if v is not None]
        )

        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )
        try:
            eval_result = evaluator.evaluate_strings(
                # The models response
                prediction=company_str,
                # The actual answer
                reference=true_answer,
                # The question asked
                input=self.question_asked,
            )
            return int(eval_result["score"])
        except ValueError as e:
            print(f"Evaluation error: {e}")
            raise
