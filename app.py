import os
from typing import TypedDict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


app = FastAPI()


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_output_tokens=5000
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class TravelState(TypedDict):
    destination: str
    days: str
    budget: str
    people: str
    interests: str
    hotel_preference: str
    plan: str
    activities: str
    hotels: str
    budget_check: str
    review: str
    final_plan: str


# ============================================================
# USER REQUEST
# ============================================================

class TravelRequest(BaseModel):
    destination: str
    days: str
    budget: str
    people: str
    interests: str
    hotel_preference: str


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: TravelState):

    prompt = f"""
You are the main Travel Planner.

Plan a trip using these details:

Destination: {state["destination"]}
Number of days: {state["days"]}
Budget: {state["budget"]}
Number of people: {state["people"]}
Interests: {state["interests"]}
Hotel preference: {state["hotel_preference"]}

Create a basic travel strategy.

Include:

1. Type of trip
2. Important planning considerations
3. Suggested budget distribution
4. Suitable activities
5. Important travel considerations

Do not claim that hotel prices or availability are live.
"""

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }


# ============================================================
# ACTIVITY PLANNER NODE
# ============================================================

def activities_node(state: TravelState):

    prompt = f"""
You are the Activity Planner.

Plan activities for this trip.

Destination: {state["destination"]}
Number of days: {state["days"]}
Number of people: {state["people"]}
Interests: {state["interests"]}

Create a realistic day-by-day activity plan.

For each day include:

- Places to visit
- Activities
- Suggested order
- Approximate time needed
- Useful travel tips

Do not put too many activities into one day.

Do not claim that prices or availability are live.
"""

    response = llm.invoke(prompt)

    return {
        "activities": response.content
    }


# ============================================================
# HOTEL NODE
# ============================================================

def hotels_node(state: TravelState):

    prompt = f"""
You are the Accommodation Planner.

Destination: {state["destination"]}
Number of days: {state["days"]}
Budget: {state["budget"]}
Number of people: {state["people"]}
Hotel preference: {state["hotel_preference"]}

Suggest suitable accommodation areas and hotel options.

IMPORTANT:

Do NOT claim that prices or rooms are currently available.

Tell the user to verify current prices and availability
before booking.

Mention these booking websites:

Booking.com
https://www.booking.com/

Agoda
https://www.agoda.com/

Explain that the user can visit these websites to check
current prices and availability.
"""

    response = llm.invoke(prompt)

    return {
        "hotels": response.content
    }


# ============================================================
# BUDGET NODE
# ============================================================

def budget_node(state: TravelState):

    prompt = f"""
You are the Budget Checker.

Review this proposed trip.

Destination:
{state["destination"]}

Number of days:
{state["days"]}

Budget:
{state["budget"]}

Number of people:
{state["people"]}

Initial Plan:
{state["plan"]}

Activities:
{state["activities"]}

Accommodation:
{state["hotels"]}

Check whether the trip appears realistic within the stated
budget.

Divide the budget into:

- Accommodation
- Food
- Local transportation
- Activities
- Miscellaneous/emergency

Do not pretend that you know live prices.

If exact prices are uncertain, clearly tell the user to
verify current prices.
"""

    response = llm.invoke(prompt)

    return {
        "budget_check": response.content
    }


# ============================================================
# REVIEWER NODE
# ============================================================

def reviewer_node(state: TravelState):

    prompt = f"""
You are the Travel Plan Reviewer.

Review the proposed trip.

Destination:
{state["destination"]}

Initial Plan:
{state["plan"]}

Activities:
{state["activities"]}

Hotels:
{state["hotels"]}

Budget Check:
{state["budget_check"]}

Look for:

- Too many activities in one day
- Unrealistic schedules
- Budget problems
- Missing rest time
- Poor ordering of locations
- Accommodation problems
- Missing travel considerations

Give clear corrections for the final planner.

Do not create the final itinerary yet.
"""

    response = llm.invoke(prompt)

    return {
        "review": response.content
    }


# ============================================================
# FINALIZER NODE
# ============================================================

def finalizer_node(state: TravelState):

    prompt = f"""
You are the Final Travel Planner.

Create the final travel plan using all information produced
by the previous agents.

Destination:
{state["destination"]}

Days:
{state["days"]}

Budget:
{state["budget"]}

People:
{state["people"]}

Interests:
{state["interests"]}

Hotel Preference:
{state["hotel_preference"]}

Initial Plan:
{state["plan"]}

Activities:
{state["activities"]}

Hotels:
{state["hotels"]}

Budget Check:
{state["budget_check"]}

Reviewer Feedback:
{state["review"]}

Create the final answer with these sections:

1. TRIP OVERVIEW

2. DAY-BY-DAY ITINERARY

3. ACCOMMODATION SUGGESTIONS

4. BUDGET GUIDANCE

5. TRAVEL TIPS

6. BOOKING WEBSITES

Mention:

Booking.com
https://www.booking.com/

Agoda
https://www.agoda.com/

IMPORTANT:

Do not claim that hotel prices or availability are live.

Tell the user to check the booking websites for current
prices and availability before booking.

Make the final answer practical, organized and easy to follow.

Return ONLY the travel plan as normal text.
Do not return JSON.
Do not return Python code.
Do not return dictionaries.
"""


    response = llm.invoke(prompt)

    return {
        "final_plan": response.content
    }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(TravelState)

builder.add_node("planner", planner_node)
builder.add_node("activities", activities_node)
builder.add_node("hotels", hotels_node)
builder.add_node("budget", budget_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "activities")
builder.add_edge("activities", "hotels")
builder.add_edge("hotels", "budget")
builder.add_edge("budget", "reviewer")
builder.add_edge("reviewer", "finalizer")
builder.add_edge("finalizer", END)

graph = builder.compile()


# ============================================================
# FRONTEND
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Travel Planner</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
}

.container {
    max-width: 850px;
    margin: 30px auto;
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

label {
    display: block;
    margin-top: 15px;
    margin-bottom: 6px;
    font-weight: bold;
}

input,
textarea,
select {
    width: 100%;
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 15px;
}

textarea {
    min-height: 100px;
    resize: vertical;
}

button {
    display: block;
    margin: 25px auto;
    padding: 13px 25px;
    border: none;
    border-radius: 8px;
    background: #111827;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:disabled {
    opacity: 0.6;
}

#loading {
    text-align: center;
    display: none;
    margin: 20px;
}

#result {
    white-space: pre-wrap;
    line-height: 1.6;
    background: #fafafa;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-top: 20px;
}

</style>

</head>


<body>

<div class="container">

<h1>Travel Planner</h1>

<div class="subtitle">
LangGraph Powered Travel Planning Agent
</div>


<label>Destination</label>

<input
    id="destination"
    placeholder="Example: Goa"
/>


<label>Number of Days</label>

<input
    id="days"
    type="number"
    min="1"
    placeholder="Example: 4"
/>


<label>Budget</label>

<input
    id="budget"
    placeholder="Example: 50000"
/>


<label>Number of People</label>

<input
    id="people"
    type="number"
    min="1"
    placeholder="Example: 4"
/>


<label>Interests</label>

<textarea
    id="interests"
    placeholder="Example: Beaches, food, bike rides"
></textarea>


<label>Hotel Preference</label>

<select id="hotel_preference">

<option value="Budget">
Budget
</option>

<option value="Mid-range">
Mid-range
</option>

<option value="Luxury">
Luxury
</option>

</select>


<button
    id="planButton"
    onclick="planTrip()"
>
Plan My Trip
</button>


<div id="loading">
Creating your travel plan...
</div>


<div id="result"></div>


</div>


<script>

async function planTrip() {

    const button =
        document.getElementById("planButton");

    const loading =
        document.getElementById("loading");

    const result =
        document.getElementById("result");


    const destination =
        document.getElementById("destination").value.trim();

    const days =
        document.getElementById("days").value.trim();

    const budget =
        document.getElementById("budget").value.trim();

    const people =
        document.getElementById("people").value.trim();

    const interests =
        document.getElementById("interests").value.trim();

    const hotel_preference =
        document.getElementById("hotel_preference").value;


    if (
        !destination ||
        !days ||
        !budget ||
        !people ||
        !interests
    ) {

        alert("Please fill in all fields.");

        return;
    }


    button.disabled = true;

    loading.style.display = "block";

    result.innerHTML = "";


    try {

        const response = await fetch(
            "/plan",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    destination: destination,
                    days: days,
                    budget: budget,
                    people: people,
                    interests: interests,
                    hotel_preference: hotel_preference
                })
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Something went wrong."
            );

        }


        if (data.error) {

            throw new Error(data.error);

        }


        if (
            typeof data.final_plan === "string"
        ) {

            result.textContent =
                data.final_plan;

        } else {

            result.textContent =
                JSON.stringify(
                    data.final_plan,
                    null,
                    2
                );

        }


    } catch (error) {

        result.textContent =
            "Error: " + error.message;

    } finally {

        button.disabled = false;

        loading.style.display = "none";

    }

}

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return HTML


# ============================================================
# PLAN TRIP
# ============================================================

@app.post("/plan")
async def plan_trip(request: TravelRequest):

    try:

        initial_state = {
            "destination": request.destination,
            "days": request.days,
            "budget": request.budget,
            "people": request.people,
            "interests": request.interests,
            "hotel_preference": request.hotel_preference,
            "plan": "",
            "activities": "",
            "hotels": "",
            "budget_check": "",
            "review": "",
            "final_plan": ""
        }


        result = graph.invoke(initial_state)


        final_plan = result.get(
            "final_plan",
            ""
        )


        if not isinstance(
            final_plan,
            str
        ):

            final_plan = str(
                final_plan
            )


        return {
            "final_plan": final_plan
        }


    except Exception as e:

        print(
            "ERROR:",
            repr(e)
        )

        return {
            "error": str(e)
        }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
