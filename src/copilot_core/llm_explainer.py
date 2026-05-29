import os
import json
import requests
from project_core.config import API_KEY, CHAT_MODEL

class LLMExplainer:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", API_KEY)
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("OPENROUTER_MODEL", CHAT_MODEL)

    def generate_explanation(self, data):
        """
        Generates a causal explanation for the pricing changes.
        :param data: Dictionary containing:
            - date: str
            - baseline_occ: float (AI forecast without event)
            - event_impact: float (Incremental jump from Local_Event)
            - manual_shock: float (User's manual occupancy shift)
            - final_occ: float (The resulting occupancy used for pricing)
            - comp_price_change_pct: float (Percentage shift in market)
            - final_price: float (The calculated rate)
            - revenue: float (Expected earnings)
        """
        
        # Crafting the Detailed "Reasoning" Prompt
        prompt = f"""
        Role: Senior Revenue Strategy Consultant for a City Hotel.
        Task: Analyze the results of a What-If pricing simulation and explain the 'Why' to the General Manager.

        --- DATA CONTEXT ---
        - Date: {data['date']}
        - Natural Forecasted Occupancy (Baseline): {round(data['base_occupancy']*100, 1)}%
        - AI-Detected Event Impact: {round(data['event_impact']*100, 1)}% additional demand.
        - User-Defined Demand Shock: {round(data['manual_shock']*100, 1)}% shift.
        - Final Simulated Occupancy: {round(data['final_occ']*100, 1)}%
        - Market Move: Competitor prices shifted by {data['comp_price_change_pct']}%
        - Internal Rules Triggered: {', '.join(data['logic_flags'])}
        - Final Recommended Price: ${data['final_suggested_price']}
        - Projected Revenue: ${data['projected_revenue']}

        --- INSTRUCTIONS ---
        Provide a summary that follows this logic:
        1. THE DRIVERS: Explain how the combination of the AI's baseline forecast and the 'Local Event' created the initial price pressure. 
        2. THE INTERVENTION: Explain how the user's manual shock ({data['manual_shock']*100}%) or the competitor's {data['comp_price_change_pct']}% move altered that initial AI suggestion.
        3. THE VERDICT: State the financial benefit (Revenue: ${data['projected_revenue']}) and give a specific recommendation (e.g., 'Hold rates high' or 'Release discounted inventory').
        4. If a particular rule or change is not getting applied, don't mention it in summary

        --- FORMATTING RULES ---
        - DO NOT use Markdown headers (like # or ##).
        - Use **bold** for the three section titles above.
        - Keep font size consistent.
        - **ALWAYS** use the '$' symbol for any monetary value (e.g., $150.00, $20,000).
        - Tone: Professional, data-driven, and authoritative. Avoid generic phrases like 'I hope this helps'.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
            "X-Title": "Hotel RMS PoC"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        try:
            # response = self.model.generate_content(prompt)
            # return response.text
            # completion = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt
            #         }
            #     ]
            # )
            # return completion.choices[0].message.content
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error from OpenRouter: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Strategic Insight is currently unavailable: {str(e)}"

# --- Scenario Test ---
if __name__ == "__main__":
    # Simulating a "What-If" where a user added an event and a competitor hike
    mock_data = {
        "date": "2017-09-23",
        "base_occupancy": 0.65,
        "event_impact": 0.20,      # Event added 20% more demand
        "manual_shock": 0.05,      # User manually added another 5%
        "final_occ": 0.90,         # Resulting in 90% total
        "comp_price_change_pct": 15.0, # Competitors went up 15%
        "final_suggested_price": 142.50,
        "projected_revenue": 31780.00,
        "logic_flags":["Saturday Yield (+15%)"]
    }
    
    explainer = LLMExplainer()
    print("AI STRATEGIC SUMMARY:")
    print("-" * 50)
    print(explainer.generate_explanation(mock_data))
