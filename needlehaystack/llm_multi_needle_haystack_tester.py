import asyncio
import glob
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone
from models import TechCompany
import numpy as np
from typing import List, Type, TypeVar
from evaluators import Evaluator
from llm_needle_haystack_tester import LLMNeedleHaystackTester
from providers import ModelProvider
import csv


class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.

    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
    """

    def __init__(
        self,
        *args,
        needles=[],
        model_to_test: ModelProvider = None,
        evaluator: Evaluator = None,
        print_ongoing_status=True,
        eval_set="multi-needle-eval-sf",
        **kwargs,
    ):

        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at
        designated depth percentages, effectively distributing these needles throughout the context. This method
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens.
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring
        that the total token count (context plus needles) does not exceed the maximum allowable context length,
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even
        spacing for the remaining needles based on the remaining context length. It ensures that needles are
        distributed as evenly as possible throughout the context after the first insertion.

        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.

        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(
            len(self.model_to_test.encode_text_to_tokens(needle))
            for needle in self.needles
        )

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[: context_length - total_needles_length]

        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)

        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.model_to_test.encode_text_to_tokens(".")

                # Then we iteration backwards until we find the first period
                while (
                    tokens_new_context and tokens_new_context[-1] not in period_tokens
                ):
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]

                # Insert the needle into the context at the found position
                tokens_context = (
                    tokens_context[:insertion_point]
                    + tokens_needle
                    + tokens_context[insertion_point:]
                )

                # Log
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(
                    f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens"
                )

                # Adjust depth for next needle
                depth_percent += depth_percent_interval

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.

        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.

        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    def read_context_files(self):
        context = ""
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
        for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
            with open(file, "r") as f:
                context += f.read()
        return context

    async def process_entire_corpus(self):
        """
        Processes the entire corpus of context files, encoding them to tokens and evaluating the model's performance
        with each context length and depth percentage combination.

        The method reads the context files, encodes them to tokens, and iterates through each context length and depth
        percentage combination. It then evaluates the model's performance with each combination and logs the results.

        The method also checks if a result already exists for a given context length and depth percentage combination
        and skips the evaluation if it does. This is useful for resuming testing after an interruption.

        The method saves the results and context to files if the corresponding flags are set.

        Returns:
            None

        """
        corpus_text = self.read_context_files()
        corpus_tokens = self.model_to_test.encode_text_to_tokens(corpus_text)

        # Iterate through each context length and depth percentage combination
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                start_index = 0
                # Iterate through the corpus in chunks of the specified context length
                while start_index < len(corpus_tokens):
                    end_index = start_index + context_length
                    if end_index > len(corpus_tokens):
                        end_index = len(corpus_tokens)

                    # Extract the context tokens

                    context_tokens = corpus_tokens[start_index:end_index]
                    context = self.model_to_test.decode_tokens(context_tokens)

                    # Evaluate the model's performance with the context

                    await self.evaluate_and_log(
                        context, context_length, depth_percent, start_index
                    )

                    start_index = end_index

    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.

        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.

        Returns:
            str: The context with needles inserted.
        """
        print(
            f"Generating context with length: {context_length}, depth: {depth_percent}%"
        )
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        print(f"Initial context length: {len(context)} tokens")
        # context = await self.insert_needles(context, depth_percent, context_length)
        return context

    async def extract_needle(self, prompt: str) -> List[TechCompany]:
        """
        Extracts the needle (fact) from the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            List[TechCompany]: A list of TechCompany instances extracted from the model's response.

        """
        example_needles_str = (
            ", ".join(self.example_needles)
            if self.example_needles
            else "No examples provided"
        )
        print("example_needles_str: ", example_needles_str)
        print(f"Extracting needle from model response...")
        try:
            fprompt = prompt
            response = await self.model_to_test.evaluate_model(fprompt)
            print(f"Raw response from model: {response}")

            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                companies_data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in the response")
            extracted_companies = []
            for company_data in companies_data:
                try:
                    company = TechCompany(**company_data)
                    extracted_companies.append(company)
                except Exception as e:
                    print(f"Error creating TechCompany instance: {str(e)}")
                    print(f"Problematic data: {company_data}")
        except json.JSONDecodeError:
            print("Failed to parse JSON from model response")
            extracted_companies = []
        except Exception as e:
            print(f"Error creating TechCompany instances: {str(e)}")
            extracted_companies = []

        print(f"Number of extracted companies: {len(extracted_companies)}")
        for company in extracted_companies:
            print(f"Extracted company: {company}")

        return extracted_companies

    def result_exists(self, context_length, depth_percent, start_index):
        """
        Checks if a result already exists for a given context length and depth percent combination.

        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
            start_index (int): The starting index of the context in the corpus.

        Returns:
            bool: True if a result exists, False otherwise.

        """
        results_dir = "results/"
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    if (
                        result["context_length"] == context_length
                        and result["depth_percent"] == depth_percent
                        and result.get("start_index", -1)
                        == start_index  # Use .get() with a default value
                        and result.get("version", 1) == self.results_version
                        and result["model"] == self.model_name
                    ):
                        return True
        return False

    def save_results_and_context(
        self, results, context, context_length, depth_percent, start_index
    ):
        """
        Saves the results and context to files.

        Args:
            results (dict): The results to save.
            context (str): The context to save.
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
            start_index (int): The starting index of the context in the corpus.

        Returns:
            None

        """
        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}_start_{start_index}'

        if self.save_contexts:
            if not os.path.exists("contexts"):
                os.makedirs("contexts")
            with open(f"contexts/{context_file_location}_context.txt", "w") as f:
                f.write(context)

        if self.save_results:
            if not os.path.exists("results"):
                os.makedirs("results")
            with open(f"results/{context_file_location}_results.json", "w") as f:
                json.dump(results, f, default=str)

    async def evaluate_and_log(
        self, context, context_length, depth_percent, start_index
    ):
        """
        Evaluates the model's performance with the generated context and logs the results.

        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        print(
            f"Starting evaluation for context length: {context_length}, depth: {depth_percent}%"
        )
        if self.save_results:
            print("Checking if result exists...")
            if self.result_exists(context_length, depth_percent, start_index):
                print("Result exists, skipping...")
                return

        # Go generate the required length context and place your needle statement in
        # context = await self.generate_context(context_length, depth_percent)
        print(f"Context length: {len(context)} tokens")

        test_start_time = time.time()

        # LangSmith
        ## TODO: Support for other evaluators
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator":
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(
                chain,
                context_length,
                depth_percent,
                self.model_to_test.model_name,
                self.eval_set,
                len(self.needles),
                self.needles,
                self.insertion_percentages,
            )
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            print("EVALUATOR: OpenAI Model")
            # Prepare your message to send to the model you're going to evaluate
            prompt = self.model_to_test.generate_prompt(
                context, self.retrieval_question
            )
            print(f"length of prompt: {len(prompt)}")
            # Go see if the model can answer the question to pull out your random fact
            # response = await self.model_to_test.evaluate_model(prompt)
            response = await self.extract_needle(prompt)
            # Compare the reponse to the actual needle you placed
            # score = self.evaluation_model.evaluate_response(response)
            print(f"Response: {response}")
            score = None

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

            results = {
                # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
                "model": self.model_to_test.model_name,
                "context_length": int(context_length),
                "depth_percent": float(depth_percent),
                "version": self.results_version,
                "needle": self.needle,
                "start_index": start_index,
                "model_response": response,
                "score": score,
                "test_duration_seconds": test_elapsed_time,
                "test_timestamp_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S%z"
                ),
            }

            self.testing_results.append(results)

            if self.print_ongoing_status:
                print(f"-- Test Summary -- ")
                print(f"Duration: {test_elapsed_time:.1f} seconds")
                print(f"Context: {context_length} tokens")
                print(f"Depth: {depth_percent}%")
                print(f"Score: {score}")
                print(f"Response: {response}\n")

            # context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'
            self.save_results_and_context(
                results, context, context_length, depth_percent, start_index
            )

            if self.seconds_to_sleep_between_completions:
                await asyncio.sleep(self.seconds_to_sleep_between_completions)

    async def bound_evaluate_and_log(self, sem, *args):
        print(f"Bound evaluate and log: {args}")
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        print("Running test...")
        sem = Semaphore(self.num_concurrent_requests)
        await self.process_entire_corpus()

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                print(
                    f"Testing context length: {context_length}, depth: {depth_percent}%"
                )
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needles: {self.needles}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
        self.save_companies_from_json_to_csv()

    def save_companies_to_csv(self, companies, output_filename=None):
        """
        Saves extracted companies to a CSV file.

        Args:
            companies (list): A list of extracted companies.
            output_filename (str): The output filename for the CSV file.

        Returns:
            None

        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"extracted_companies_{self.model_name}_{timestamp}.csv"

        fieldnames = [
            "name",
            "location",
            "employee_count",
            "founding_year",
            "is_public",
            "valuation",
            "primary_focus",
        ]

        with open(output_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for company in companies:
                print(f"Processing company: {company}")
                if isinstance(company, TechCompany):
                    row = {field: getattr(company, field) for field in fieldnames}
                elif isinstance(company, dict):
                    row = {k: v for k, v in company.items() if k in fieldnames}
                else:
                    row = {}
                    for field in fieldnames:
                        value = (
                            str(company).split(f"{field}=")[1].split()[0]
                            if f"{field}=" in str(company)
                            else None
                        )
                        row[field] = value.strip("'") if value else None
                print(f"Writing row: {row}")
                writer.writerow(row)

        print(f"Saved extracted companies to {output_filename}")

    def save_companies_from_json_to_csv(
        self, results_dir="results", output_filename=None
    ):
        """
        Extracts companies from JSON files and saves them to a CSV file.

        Args:
            results_dir (str): The directory containing the JSON files.
            output_filename (str): The output filename for the CSV file.

        Returns:
            None

        """
        companies = self.extract_companies_from_json_files(results_dir)
        print(f"Companies extracted: {companies}")
        self.save_companies_to_csv(companies, output_filename)
        print(f"Processed {len(companies)} companies from JSON files.")

    def extract_companies_from_json_files(self, results_dir="results"):
        all_companies = []
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    print(f"Processing file: {filename}")
                    print(f"Contents: {result}")
                    if "model_response" in result:
                        if isinstance(result["model_response"], list):
                            all_companies.extend(result["model_response"])
                        else:
                            all_companies.append(result["model_response"])
                    else:
                        print(f"No 'model_response' found in {filename}")
        print(f"Total companies extracted: {len(all_companies)}")
        return all_companies
