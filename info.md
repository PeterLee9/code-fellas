Winter 2026 Hackathon

Problem Statements

The Future Cities Institute is proposing the following three Problem Statements for the Winter 2026 Hackathon in collaboration with BRAVE. 

Problem Statement: Build a National Zoning & Land Use Data Platform: You are building a tool to help gather manicipality data, so look at each region/manicipal's official sites, one potentially useful reference: civic.band

Lead: Leia Minaker lminaker@uwaterloo.ca 
The Challenge
Across Canada, a critical piece of the housing affordability puzzle remains hidden in fragmented, somewhat inaccessible data: local land use policies and zoning regulations. While we know that restrictive zoning can drive up housing costs, limits density, and perpetuates exclusionary development patterns, there's no comprehensive way to analyze these policies at scale.

Each municipality maintains its own zoning bylaws, official plans, and land use regulations—often buried in PDFs, spread across multiple websites, or locked in formats that resist analysis. Researchers, policymakers, developers, and housing advocates need this data to understand how municipal policies affect housing supply, but accessing it requires manually visiting thousands of municipal websites and parsing tens of thousands of documents.

That's where you come in.

Your Task
Design and build a web-scraping tool and data platform that aggregates local land use policies and zoning regulations into a searchable, analyzable, open national dataset.

The tool should:
Scrape and ingest municipal zoning data from local government websites, starting with one province but designed to scale nationally. This includes zoning bylaws, official plans, permitted use tables, density regulations, parking requirements, and setback rules.
Parse and structure the data into a standardized format that allows for comparison across jurisdictions. Extract key metrics like minimum lot sizes, building heights, dwelling unit restrictions, parking mandates, and permitted housing types.
Create a searchable database and API that allows users to query zoning regulations by municipality, policy type, housing restriction, or geographic area. Make the data accessible to researchers, journalists, developers, and advocacy organizations.
Visualize the data to show patterns and outliers—where are the most restrictive zoning policies? Which municipalities allow multi-family housing? Where do parking requirements add the most cost? Build maps, dashboards, or comparison tools that make the findings accessible.
Automate updates and validation to keep the dataset current as municipalities update their bylaws. Include mechanisms to flag changes, verify data accuracy, and handle variations in how municipalities structure their regulations.

Consider including (optional)
Natural language processing to extract policy details from PDF documents, geocoding to map zoning districts, machine learning to categorize zoning types, version control to track policy changes over time, and user contribution features to crowdsource missing data or corrections.

Why Manual Scraping Isn't Enough
A user trying to analyze zoning policies today faces impossible challenges:
Fragmentation: Thousands of municipalities each maintain their own websites, document structures, and data formats. There's no central repository.
Inaccessibility: Zoning bylaws are often published as scanned PDFs, requiring OCR and parsing. Permitted use tables may be buried in appendices or split across multiple documents.
Inconsistency: Municipalities use different terminology, classification systems, and regulatory frameworks. What one city calls "R-2" zoning might be "Residential Low Density" elsewhere.
Scale: With ~4-5000 municipalities in Canada, manual data collection would take years. Automated scraping and standardization is the only viable path to comprehensive analysis.
Currency: Bylaws change regularly. A one-time data collection effort quickly becomes outdated without automated monitoring and updates.

That's why your tool must combine intelligent web scraping with data standardization and validation—to transform scattered municipal documents into a unified, queryable dataset that can finally reveal the true impact of local land use policies on housing affordability across the country.

Technical Considerations
Scraping Strategy: Handle diverse website structures, PDF extraction (OCR where needed), and rate limiting to avoid overwhelming municipal servers.
Data Schema: Design a flexible schema that captures common zoning parameters while accommodating jurisdictional variations.
Quality Assurance: Implement validation rules, flag anomalies, and provide mechanisms for human review of ambiguous data.
Scalability: Start with one province as proof-of-concept, but architect for national expansion across all provinces and territories.
Open Access: Make the dataset publicly available under an open license. Provide clear documentation, API access, and tools for analysis.

Impact
This platform would enable:
Academic researchers to study the relationship between zoning restrictions and housing costs at unprecedented scale
Policymakers to benchmark their regulations against comparable municipalities and identify reform opportunities
Housing advocates to pinpoint exclusionary policies and build evidence-based cases for reform
Developers and builders to identify markets where zoning enables or restricts different housing types
Journalists to investigate patterns of restrictive zoning and its impact on affordability and equity

By making this data accessible, you'll create the foundation for evidence-based housing policy reform across Canada—transforming how we understand and address the zoning barriers that drive housing unaffordability.

Deep Agents & LangChain:  Building AI Agents for Hackathon Projects
Github Demo: https://github.com/CatMizu/demo-agent

Peter27 <lde492@gmail.com>
1:16 PM (50 minutes ago)
to minjia.zhu

Hi,

I'm Peter from Code Fellas team. I have a couple of questions regarding the hackathon:
1. Is there a link for submission?
2. Is there a meeting that online participants can join to ask question if anything comes up?
3. For problem statement 1, is it okay if we build a prototype using top cities in Canada, not every single cities?

Thank you!

Minjia Zhu
1:32 PM (34 minutes ago)
to me

Hi Peter,

1. Will share the google form submission link soon. 
2. I will open a meeting for demo viewing party, will share the link with the submission link too
3. Yes, for sure, i dont think its possible to get every single city in Canada, so you can def use a few cities as an example, and demonstrate how your work can be applied to other cities. 

Best regards,
Minjia