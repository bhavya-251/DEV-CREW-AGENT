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
    temperature=0.7,
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
# USER INPUT
# ============================================================

class TravelRequest(BaseModel):
    destination: str
    days: str
    budget: str
    people: str
    interests: str
    hotel_preference: str


# ============================================================
# PLANNER
# ============================================================

def planner_node(state: TravelState):

    prompt = f"""
You are the Travel Planner in a travel planning system.

Create a basic travel strategy from these details:

Destination: {state["destination"]}
Number of days: {state["days"]}
Budget: {state["budget"]}
Number of people: {state["people"]}
Interests: {state["interests"]}
Hotel preference: {state["hotel_preference"]}

Identify:
1. The type of trip
2. Important planning considerations
3. How the budget should be divided
4. What kind of activities would suit the traveller

Do not invent live prices or availability.
"""

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }


# ============================================================
# ACTIVITY PLANNER
# ============================================================

def activities_node(state: TravelState):

    prompt = f"""
You are the Activity Planner.

Plan activities for this trip.

Destination: {state["destination"]}
Days: {state["days"]}
Number of people: {state["people"]}
Interests: {state["interests"]}

Create a realistic day-by-day activity plan.

For every day include:
- Places to visit
- Activities
- Suggested order
- Approximate time needed
- Useful travel tips

Do not claim that prices or availability are live.
"""

    response = llm.invoke(prompt)

    return {
        "activities": response.content
    }


# ============================================================
# HOTEL RECOMMENDER
# ============================================================

def hotels_node(state: TravelState):

    prompt = f"""
You are the Accommodation Planner.

Destination: {state["destination"]}
Number of days: {state["days"]}
Budget: {state["budget"]}
Number of people: {state["people"]}
Hotel preference: {state["hotel_preference"]}

Suggest suitable types of accommodation and, where you
are confident, examples of hotels or accommodation areas.

IMPORTANT:
Do NOT claim that prices or rooms are currently available.

Give the user these websites where they can check current
prices and availability:

Booking.com:
https://www.booking.com/

Agoda:
https://www.agoda.com/

Make it clear that the user should check the websites
before booking.
"""

    response = llm.invoke(prompt)

    return {
        "hotels": response.content
    }


# ============================================================
# BUDGET CHECKER
# ============================================================

def budget_node(state: TravelState):

    prompt = f"""
You are the Budget Checker.

Review this proposed trip.

Destination: {state["destination"]}
Days: {state["days"]}
Budget: {state["budget"]}
People: {state["people"]}

Initial planning:
{state["plan"]}

Activities:
{state["activities"]}

Accommodation suggestions:
{state["hotels"]}

Determine whether the proposed trip appears realistic
within the stated budget.

Break the budget into categories such as:
- Accommodation
- Food
- Local transport
- Activities
- Emergency/miscellaneous

Do NOT pretend to know live prices.

If exact prices are unavailable, clearly say that the user
should verify current prices before booking.
"""

    response = llm.invoke(prompt)

    return {
        "budget_check": response.content
    }


# ============================================================
# REVIEWER
# ============================================================

def reviewer_node(state: TravelState):

    prompt = f"""
You are the Travel Plan Reviewer.

Review the following proposed trip:

Destination:
{state["destination"]}

Plan:
{state["plan"]}

Activities:
{state["activities"]}

Hotels:
{state["hotels"]}

Budget Check:
{state["budget_check"]}

Find problems such as:
- Too many activities in one day
- Unrealistic travel schedule
- Budget concerns
- Missing rest time
- Poor activity ordering
- Accommodation issues

Then give specific corrections.

Do not generate the final itinerary yet.
"""

    response = llm.invoke(prompt)

    return {
        "review": response.content
    }


# ============================================================
# FINALIZER
# ============================================================

def finalizer_node(state: TravelState):

    prompt = f"""
You are the Final Travel Planner.

Create the final travel plan using all previous information.

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

Hotel preference:
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

Create a clear final answer containing:

1. Trip Overview
2. Day-by-Day Itinerary
3. Accommodation Suggestions
4. Budget Guidance
5. Travel Tips
6. Hotel/Booking Websites

Use these websites:

Booking.com:
https://www.booking.com/

Agoda:
https://www.agoda.com/

IMPORTANT:
Do not claim that hotel prices or availability are live.
Tell the user to check the websites for current prices
and availability.

Make the final plan practical and easy to follow.
"""

    response = llm.invoke(prompt)

    return {
        "final_plan": response.content
    }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(TravelState)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "activities",
    activities_node
)

builder.add_node(
    "hotels",
    hotels_node
)

builder.add_node(
    "budget",
    budget_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.add_node(
    "finalizer",
    finalizer_node
)


builder.add_edge(
    START,
    "planner"
)

builder.add_edge(
    "planner",
    "activities"
)

builder.add_edge(
    "activities",
    "hotels"
)

builder.add_edge(
    "hotels",
    "budget"
)

builder.add_edge(
    "budget",
    "reviewer"
)

builder.add_edge(
    "reviewer",
    "finalizer"
)

builder.add_edge(
    "finalizer",
    END
)


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
}

</style>

</head>


<body>

<div class="container">

<h1>Travel Planner</h1>

<div class="subtitle">
Plan your trip using a LangGraph-powered travel assistant
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
    placeholder="Example: ₹20000"
/>


<label>Number of People</label>

<input
    id="people"
    type="number"
    min="1"
    placeholder="Example: 2"
/>


<label>Interests</label>

<textarea
    id="interests"
    placeholder="Example: Beaches, sightseeing, food, adventure"
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
        document.getElementById(
            "planButton"
        );

    const loading =
        document.getElementById(
            "loading"
        );

    const result =
        document.getElementById(
            "result"
        );


    const destination =
        document.getElementById(
            "destination"
        ).value.trim();


    const days =
        document.getElementById(
            "days"
        ).value.trim();


    const budget =
        document.getElementById(
            "budget"
        ).value.trim();


    const people =
        document.getElementById(
            "people"
        ).value.trim();


    const interests =
        document.getElementById(
            "interests"
        ).value.trim();


    const hotel_preference =
        document.getElementById(
            "hotel_preference"
        ).value;


    if (
        !destination ||
        !days ||
        !budget ||
        !people ||
        !interests
    ) {

        alert(
            "Please fill in all fields."
        );

        return;
    }


    button.disabled = true;

    loading.style.display =
        "block";

    result.innerHTML = "";


    try {

        const response =
            await fetch(
                "/plan",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            destination,
                            days,
                            budget,
                            people,
                            interests,
                            hotel_preference
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Something went wrong."
            );
        }


        result.textContent =
            data.final_plan;


    } catch (error) {

        result.textContent =
            "Error: " +
            error.message;

    } finally {

        button.disabled = false;

        loading.style.display =
            "none";
    }
}

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return HTML


# ============================================================
# PLAN TRIP
# ============================================================

@app.post("/plan")
async def plan_trip(
    request: TravelRequest
):

    try:

        initial_state = {

            "destination":
                request.destination,

            "days":
                request.days,

            "budget":
                request.budget,

            "people":
                request.people,

            "interests":
                request.interests,

            "hotel_preference":
                request.hotel_preference,

            "plan":
                "",

            "activities":
                "",

            "hotels":
                "",

            "budget_check":
                "",

            "review":
                "",

            "final_plan":
                ""
        }


        result =
            graph.invoke(
                initial_state
            )


        return {
            "final_plan":
                result["final_plan"]
        }


    except Exception as e:

        print(
            "ERROR:",
            repr(e)
        )

        return {
            "error":
                str(e)
        }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
