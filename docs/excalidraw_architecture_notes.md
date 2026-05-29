# Excalidraw Architecture Board Notes

Open this board in the VS Code Excalidraw extension:

- `docs/hotel_rms_architecture.excalidraw`

How to use it:

1. Start with the top section **01 Overview - Main Parts**.
2. Zoom or pan into the drill-down sections below:
   - **02 Data, PMS, OTB, Cancellation**
   - **03 Forecasting And Model Governance**
   - **04 Market Feed And Pricing Engine**
   - **05 Local Intel, AI, Scenario Copilot**
   - **06 Streamlit Views And Outputs**
   - **07 Governance, Docs, Tests**
3. Use the original Mermaid file as the complete technical inventory:
   - `docs/project_functionality_flowchart.md`

Compatibility note:

- The board uses plain rectangles as section containers instead of native Excalidraw frame elements. This is intentional because some VS Code Excalidraw extension versions fail to open files that contain newer `frame` element metadata.

Recommended presentation flow:

1. Explain the overview frame first.
2. Drill into Forecasting if the audience is data science oriented.
3. Drill into Pricing + Local Intel if the audience is hotel revenue management oriented.
4. End with Governance to show why the system is inspectable rather than a black box.
