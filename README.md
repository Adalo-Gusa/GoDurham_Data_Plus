GoDurham – Bus Stop Inventory & Bayesian Cleaning Priority Modeling
Overview
This repository contains the modeling, data collection, and deployment workflow for the GoDurham Data+ 2026 Project.

Project Goals:

Automated Inventory: Create an up-to-date desktop inventory of GoDurham's 839+ bus stops using web scraping and AI, minimizing manual field visits.

Data-Driven Cleaning: Replace manual cleaning schedules with a Bayesian statistical model that predicts cleaning needs based on infrastructure, ridership, complaints, demographics, and spatial data.

🛠️ Key Pipeline Features
Automated Data Collection: A custom Python scraper pulls recent (Jan 2025+) Google Maps Street View imagery for active bus stops, using fallback logic for obscured locations.

AI Vision Classification: Uses Gemini 3.5 Flash (via Google Cloud Vertex AI) to detect and classify amenities like shelters, benches, trash cans, and ADA-compliant surfaces, outputting directly to strict JSON schemas.

Live ArcGIS Deployment: A Streamlit UI allows city technicians to review AI classifications and instantly sync edits to the City of Durham's live ArcGIS map server.

Field Verification Protocol: Includes a manual QA/QC guide for the ~33 stops with out-of-date or obstructed camera views.

📊 Analytical Workflow
Please review the Jupyter Notebooks in the following order:

1. Bayesian_Modeling.ipynb
Develops a Bayesian hierarchical Zero-Inflated Negative Binomial (ZINB) model to predict bus stop cleaning needs.

Key Steps: Data prep, PyMC model fitting, convergence diagnostics, and posterior predictions.

2. Results_and_Spatial_Analysis.ipynb
Interprets model outputs and maps the results.

Key Steps: Infrastructure summaries, equity/demographic analysis, hot-spot mapping, and defining high/low-priority cleaning tiers.

💻 Tech Stack
AI/ML: Google Cloud Vertex AI, Gemini 3.5 Flash, PyMC, ArviZ

Geospatial: ArcGIS Python API, GeoPandas

Data & UI: Python, pandas, Streamlit, Matplotlib

🔒 Data Availability
To protect internal city records and privacy, the raw datasets (complaint histories, granular ridership) are not included publicly. However, the complete code and structural data schemas are provided so the analysis can be reproduced with equivalent datasets.