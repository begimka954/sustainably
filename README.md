# 🌍 Sustainably - AI-Powered Social Impact Reporting

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sustainably.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20Powered-green.svg)](https://openai.com/)

**Transform your social impact data into compelling funding reports in minutes, not months.**

Sustainably is an AI-powered platform designed specifically for NGOs and Social Enterprises to automate the creation of professional social impact reports. Say goodbye to expensive consultants and time-consuming manual reporting processes.

## 🚀 Live Demo

👉 **[Try Sustainably Now](https://sustainably.streamlit.app)**

## ✨ Key Features

### 🤖 **AI-Powered Analysis**
- Intelligent data interpretation using GPT models
- Automated insight generation
- Smart recommendations based on your data patterns
- Framework-specific analysis (UN SDGs, IRIS+, CIW)

### 📊 **Multi-Framework Support**
- **UN Sustainable Development Goals** - Perfect for international funding
- **IRIS+ Impact Measurement** - Ideal for impact investors
- **Canadian Index of Wellbeing** - Tailored for Canadian organizations

### 📁 **Flexible Data Input**
- CSV, Excel, and JSON file support
- Automatic data quality assessment
- Real-time data preview and validation
- Sample data templates included

### 📑 **Professional Report Generation**
- Executive summaries for board meetings
- Detailed reports for comprehensive review
- Funder presentations ready for investment pitches
- Multiple export formats (PDF, Excel, PowerPoint)

### 🎯 **User-Centric Design**
- Intuitive 4-step workflow
- No coding knowledge required
- Beautiful, modern interface
- Mobile-responsive design

## 🛠️ Quick Start

### Option 1: Use the Live App (Recommended)
1. Visit [sustainably.streamlit.app](https://sustainably.streamlit.app)
2. Upload your impact data
3. Select your reporting framework
4. Generate your AI-powered report!

### Option 2: Run Locally

#### Prerequisites
- Python 3.8 or higher
- OpenAI API key (optional, for AI features)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sustainably.git
cd sustainably
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 🔑 AI Configuration

### Getting an OpenAI API Key
1. Visit [OpenAI's website](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Add it to the sidebar in the Sustainably app

### Cost Estimation
- **GPT-3.5 Turbo**: ~$0.002 per report
- **GPT-4**: ~$0.06 per report

*Note: The app works without an API key using simulated AI analysis.*

## 📋 Supported Data Formats

### CSV Files
```csv
Date,Beneficiaries_Served,Programs_Delivered,SDG_Education,Funding_Received
2024-01-01,150,5,80,25000
2024-02-01,180,6,95,30000
```

### Excel Spreadsheets
- .xlsx and .xls formats supported
- Multiple sheets automatically detected
- Headers automatically identified

### JSON Data
```json
{
  "impact_data": [
    {"month": "Jan", "beneficiaries": 150, "programs": 5},
    {"month": "Feb", "beneficiaries": 180, "programs": 6}
  ]
}
```

## 🎯 Use Cases

### For NGOs
- **Grant Applications**: Generate compelling impact narratives
- **Annual Reports**: Create comprehensive year-end summaries
- **Donor Relations**: Provide regular impact updates
- **Board Meetings**: Present clear performance metrics

### For Social Enterprises
- **Impact Investing**: Demonstrate social returns to investors
- **ESG Reporting**: Meet environmental and social governance requirements
- **Stakeholder Communication**: Keep stakeholders informed of progress
- **Performance Tracking**: Monitor impact KPIs continuously

### For Foundations
- **Grantee Reporting**: Standardize impact reporting across portfolio
- **Due Diligence**: Assess potential funding recipients
- **Impact Assessment**: Measure collective portfolio impact
- **Strategic Planning**: Use data insights for funding decisions

## 🏗️ Technical Architecture

### Frontend
- **Streamlit**: Python-based web app framework
- **Plotly**: Interactive data visualizations
- **Custom CSS**: Beautiful, responsive design

### AI Integration
- **OpenAI GPT Models**: Natural language generation
- **Custom Prompts**: Framework-specific analysis
- **Intelligent Parsing**: Structured output generation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **Multiple Format Support**: CSV, Excel, JSON
- **Data Quality Checks**: Automatic validation and cleaning

## 🛣️ Roadmap

### Phase 1: MVP ✅
- [x] Core platform development
- [x] Basic AI integration
- [x] Multi-framework support
- [x] Report generation

### Phase 2: Enhancement 🚧
- [ ] Advanced AI models fine-tuning
- [ ] User authentication system
- [ ] Data persistence and history
- [ ] Advanced visualizations

### Phase 3: Scale 📋
- [ ] Multi-tenant architecture
- [ ] API for third-party integrations
- [ ] White-label solutions
- [ ] Enterprise features

### Phase 4: Expansion 🌟
- [ ] Additional reporting frameworks
- [ ] Predictive analytics
- [ ] Collaborative features
- [ ] Mobile application

## 🤝 Contributing

We welcome contributions from the community! 



## 🙏 Acknowledgments

- **OpenAI** for providing the AI capabilities
- **Streamlit** for the amazing web app framework
- **The Social Impact Community** for inspiration and feedback
- **Open Source Contributors** who make projects like this possible



---

<div align="center">

**Built with ❤️ for the social impact community**

[Website](https://sustainably.ai) • [Demo](https://sustainably.streamlit.app) • [Documentation](https://docs.sustainably.ai) • [Community](https://discord.gg/sustainably)

</div>