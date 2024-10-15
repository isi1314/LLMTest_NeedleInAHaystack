from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI
from typing import List, Optional
from llm_needle_haystack_tester import LLMNeedleHaystackTester
from llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from providers import Anthropic, ModelProvider, OpenAI, Cohere

load_dotenv()


@dataclass
class CommandArgs:

    provider: str = "openai"
    evaluator: str = "openai"
    model_name: str = "gpt-4-turbo"
    evaluator_model_name: Optional[str] = "gpt-4-turbo"
    needle: Optional[str] = (
        "\nRyoshi, based in Neo Tokyo, Japan, is a private quantum computing firm founded in 2031, currently valued at $8.7 billion with 1,200 employees focused on quantum cryptography.\n"
    )
    haystack_dir: Optional[str] = "haystack"
    retrieval_question: Optional[str] = (
        "Extract information about ALL technology companies mentioned in the text. If all properties are not available, provide what is present."
    )
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 5000
    context_lengths_num_intervals: Optional[int] = 2  # 35
    context_lengths: Optional[list[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 2  # 35
    document_depth_percents: Optional[list[int]] = None
    document_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = False
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    example_needles: Optional[List[str]] = field(
        default_factory=lambda: [
            "Ryoshi, based in Neo Tokyo, Japan, is a private quantum computing firm founded in 2031, currently valued at $8.7 billion with 1,200 employees focused on quantum cryptography.",
            "Ryoshi is a private quantum computing firm founded in 2031, currently valued at $8.7 billion with 1,200 employees focused on quantum cryptography.",
            "Ryoshi is a private quantum computing firm founded in 2031.",
            "NeuraNet, a private biotech firm based in Atlantis City, Pacific Ocean, was founded in 2022, has 950 employees, and specializes in neural interface technologies, valued at $2.6 billion.",
        ]
    )
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: list[str] = field(
        default_factory=lambda: [
            " Figs are one of the secret ingredients needed to build the perfect pizza. ",
            " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ",
            " Goat cheese is one of the secret ingredients needed to build the perfect pizza. ",
        ]
    )


def get_model_to_test(args: CommandArgs) -> ModelProvider:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.

    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.

    Returns:
        ModelProvider: An instance of the specified model provider class.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name)
        case "anthropic":
            return Anthropic(model_name=args.model_name)
        case "cohere":
            return Cohere(model_name=args.model_name)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")


def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.

    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.

    Returns:
        Evaluator: An instance of the specified evaluator class.

    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(
                model_name=args.evaluator_model_name,
                question_asked=args.retrieval_question,
                true_answer=args.needle,
            )
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")


def main():
    """
    The main function to execute the testing process based on command line arguments.

    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)

    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else:
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    results = tester.start_test()
    for result in results:
        print(f"Result: {result}")
    print(f"Total results: {len(results)}")


if __name__ == "__main__":
    main()
