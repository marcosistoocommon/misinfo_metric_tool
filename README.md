# Misinformation Metric Tool


A research prototype for the analysis and detection of misinformation campaigns on social media.

This project was developed as part of the Bachelor's Thesis **"Diseño y Desarrollo de un Sistema para el Análisis y Detección de Campañas de Desinformación en Redes Sociales"**. The project explores the development of a metric for estimating misinformation risk by combining multiple signals extracted from social media content and its surrounding context.

The objective is to provide a quantitative indicator that can support the analysis, comparison, and prioritization of potentially misleading social media posts. Rather than relying on a single characteristic, the system combines different analytical components, including linguistic patterns, tone, contextual information, and claim verification.

## Table of Contents

- [About the Thesis](#about-the-thesis)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Tool](#using-the-tool)
- [Misinformation Metric](#misinformation-metric)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)


## About the Thesis

The project addresses the growing impact of misinformation on social media and the difficulty of systematically assessing its potential risk. The lack of standardized approaches for measuring misinformation makes it difficult to compare individual posts and determine which content should receive greater attention.

The thesis proposes the design and development of a system that combines multiple analytical techniques into a unified misinformation risk metric. By assigning a quantitative score to analyzed content, the system aims to facilitate the comparison and prioritization of potentially misleading posts and provide a basis for further misinformation analysis.

### Thesis Information

* **Title:** Diseño y Desarrollo de un Sistema para el Análisis y Detección de Campañas de Desinformación en Redes Sociales
* **Degree:** Bachelor's Degree in Telecommunications Technologies and Services Engineering
* **Year:** 2026
* **Supervisor:** Sonia Solera Cotanilla
* **Thesis:** [Read the full thesis](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/memoria.pdf)


## Project Overview

The system analyzes social media content using several complementary components:

* **Claim analysis and verification** to identify and assess factual claims.
* **Pattern analysis** to detect signals associated with misinformation, including bias, propaganda, fallacies, emotion, hate speech, and violence.
* **Tone analysis** to evaluate the tone of the analyzed content.
* **Context analysis** to incorporate information surrounding the original post.
* **Misinformation scoring** to aggregate the different analytical signals into a final risk score.

![System Architecture](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/mermaid-diagram.png)

The resulting score is intended to serve as an analytical indicator of potential misinformation risk. It should not be interpreted as an absolute determination of whether a statement is objectively true or false.

## Key Features

* Analysis of social media posts and their context.
* Claim extraction and verification.
* Detection of linguistic and rhetorical patterns associated with misinformation.
* Tone analysis.
* Web-based prototype for interactive analysis.
* Support for English and Spanish languages.
* Modular architecture allowing individual analysis components to be evaluated independently.

## Repository Structure

```text
.
├── Claims/              # Claim extraction and analysis
├── ClaimeAI/            # Claim verification components
├── Context/             # Context retrieval and analysis
├── Patterns/            # Linguistic and rhetorical pattern detectors
├── Tone/                # Tone analysis
├── docs/               # Thesis documentation
├── app.py               # Flask web application
├── misinfo_value.py     # Misinformation score aggregation
├── translate.py         # Translation and language-processing utilities
├── requirements.txt     # Python dependencies
└── .env                 # Local environment configuration 
```

## Installation

### Requirements

Before installing the project, make sure you have:

* Python 3.x installed.
* A Git installation.
* API keys for the external services used by the project.
* An authenticated X/Twitter account for data collection through `twscrape`.

### 1. Clone the repository

```bash
git clone https://github.com/marcosistoocommon/misinfo_metric_tool.git
cd misinfo_metric_tool
```

### 2. Create a virtual environment

It is recommended to use a virtual environment to isolate the project's dependencies.

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade `pip` and install the dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

The project requires API credentials for several external services. Create a `.env` file in the root directory of the project.

The following variables are required:

```env
TANBIH_API_KEY=your_tanbih_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Replace each placeholder with the corresponding API key.

Do not commit the `.env` file to the repository. API keys and other credentials should be kept private.

### 5. Configure X/Twitter scraping

The project uses [`twscrape`](https://github.com/vladkens/twscrape) and [`scweet`](https://github.com/Alir3z4/scweet) to obtain data from X/Twitter. This requires configuring authenticated X/Twitter accounts and generating the required `cookies.json` file.

Follow the official documentation for the current account and cookie setup procedure:

**twscrape documentation:** https://github.com/vladkens/twscrape

**scweet documentation:** https://github.com/Alir3z4/scweet

After completing the setup, ensure that the required `cookies.json` file is available in the location expected by the project before running the web application or analysis pipeline.

> **Important:** X/Twitter scraping depends on the availability and behavior of the platform and the libraries. Changes to X/Twitter authentication or platform restrictions may require changes to the scraping configuration.

## Running the Application

Once the installation and configuration steps are complete, start the Flask web prototype with:

```bash
python app.py
```

The application will display the local address in the terminal. Open this address in a web browser to access the interface.

The web prototype allows users to submit an X/Twitter post for analysis and view the resulting misinformation risk assessment and its individual analytical components.

![Home screen screenshot](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/in.png)

![Loading screen screenshot](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/in2.png)

![Results screen screenshot](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/in3.png)

Also, individual testing of the different analysis components can be performed. To do this, run the individual `component.py` script for the desired component, providing the required input parameters. For example:

```bash
python Patterns/emotion.py
```

## Using the Tool

To use the tool, follow these steps:

1. Submit an X/Twitter post for analysis. The post can be submitted by providing the URL of the post, with the format `https://twitter.com/<username>/status/<post_id>`.

2. View the resulting misinformation risk assessment and its individual analytical components.

## Misinformation Metric

The final misinformation risk score is calculated by combining four main components:

- Pattern analysis: 40%
- Tone analysis: 20%
- Context analysis: 10%
- Claim verification: 30%

The resulting score provides an aggregated indication of the potential misinformation risk associated with the analyzed content.

The metric is designed to combine multiple signals rather than relying on a single classifier or characteristic.

For more details on the methodology and calculation of the misinformation metric, please refer to the thesis documentation in the [`docs`](https://github.com/marcosistoocommon/misinfo_metric_tool/tree/main/docs) directory.

## Future Work

The current implementation is a research prototype and has several limitations. Future work could focus on:

- Improving the resource management and performance of the system to handle larger datasets and more complex analyses.  
- Developing multimodal analysis capabilities to incorporate images, videos, and other media types.
- Including the development of own models to increase accuracy and reduce dependency on external services.
- Using machine learning techniques to improve the accuracy of the misinformation metric and adapt to evolving patterns of misinformation.


## Citation

If you use this project or its methodology in academic work, please cite the associated thesis:

> Marcos Pérez. *Diseño y Desarrollo de un Sistema para el Análisis y Detección de Campañas de Desinformación en Redes Sociales*. Bachelor's Degree in Telecommunications Technologies and Services Engineering, 2026.

The full thesis is available in the [`docs`](./docs/) directory or [online](https://github.com/marcosistoocommon/misinfo_metric_tool/blob/main/docs/memoria.pdf).

## License
This project is licensed under the **MIT** License. 
