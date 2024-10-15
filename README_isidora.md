# LLM Needle Haystack Tester

## Overview
This project implements a system for testing Large Language Models (LLMs) using a "needle in a haystack" approach. It includes a tester class, an OpenAI provider for interacting with OpenAI's API, and a Pydantic model for representing technology companies.

### Setup Virtual Environment - Alternative option

Follow the directions for [Micromamba installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)._

Set up a mamba environment:
```sh
micromamba create -n haystack python=3.11
```

Activate your new environment:
```sh
micromamba activate haystack
```

Install the `poetry` dependency management tool:
```sh
pip install poetry
```

```sh
poetry init
```

From the repo's root directory, install the dependencies using poetry:
```sh
poetry install
```

### Environment Variables -  .env file
You will need to add a .env file in the root of the repository. Below are example default values:
- `NIAH_MODEL_API_KEY` - API key for interacting with the model. Depending on the provider, this gets used appropriately with the correct sdk.
- `NIAH_EVALUATOR_API_KEY` - API key to use if `openai` evaluation strategy is used.


## Key Components

## TechCompany Model

The `TechCompany` model is defined in `models.py` and includes the following fields:

- `name`: The full name of the technology company (Optional[str])
- `location`: City and country where the company is headquartered (Optional[str])
- `employee_count`: Total number of employees (Optional[int])
- `founding_year`: Year the company was established (Optional[int])
- `is_public`: Whether the company is publicly traded or privately held (Optional[bool])
- `valuation`: Company's valuation in billions of dollars (Optional[float])
- `primary_focus`: Main area of technology or industry the company focuses on (Optional[str])

## Usage

The `TechCompany` model is used within the LLMNeedleHaystackTester to structure and validate information extracted by the language models being tested.