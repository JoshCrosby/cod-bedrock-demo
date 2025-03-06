# Chain of Draft (CoD) - Bedrock Implementation Demo

This repository contains an implementation of the Chain of Draft (CoD) methodology using Amazon Bedrock. CoD is a technique designed to improve token efficiency in language models by using a draft-then-refine approach.

## About Chain of Draft

Chain of Draft is a token-efficient prompting strategy that leverages a simpler draft model to generate initial content that is then refined by a more powerful model. This approach can significantly reduce token usage and cost while maintaining output quality.

The comprehensive explanation of this methodology is available in the included whitepaper: "ChainofDraft Whitepaper.pdf".

## Key Features

- Implementation of the Chain of Draft methodology with Amazon Bedrock
- Comparative analysis between standard prompting and CoD
- Performance metrics including token usage, time, and cost
- Support for Claude and Anthropic models

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- AWS account with access to Amazon Bedrock

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/JoshCrosby/cod-bedrock-demo.git
   cd cod-bedrock-demo
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Activate the Poetry environment:
   ```
   poetry shell
   ```

## Configuration

Ensure your AWS credentials are properly configured with access to Amazon Bedrock. You can set them up using:

```
aws configure
```

## Running the Code

The main script (`main.py`) demonstrates the CoD approach with Amazon Bedrock:

```
poetry run python main.py
```

This will:
1. Run a series of test prompts using both standard prompting and Chain of Draft
2. Compare performance metrics (token usage, time, cost)
3. Save results to `comparison_results.json`

## Understanding the Results

The results are documented in the `RESULTS.md` file, which includes:

- Detailed performance metrics for each approach
- Token usage analysis
- Cost comparison
- Execution time metrics

### Key Findings

As shown in the results, the Chain of Draft approach typically:
- Reduces token usage by leveraging a simpler draft model
- Maintains comparable output quality to standard prompting
- Offers cost savings for inference operations
- May introduce a slight latency increase due to the two-step process

## Visualization

The repository includes a visualization (`Figure_1.png`) that demonstrates the comparative performance between standard prompting and the Chain of Draft approach.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms included in the LICENSE file.

## Citation

If you use this implementation in your research, please cite the Chain of Draft whitepaper.

## Substack Article
https://open.substack.com/pub/joshcrosby/p/rag-vs-cot-vs-cod
