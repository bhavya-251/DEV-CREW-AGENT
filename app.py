import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI()


# ============================================================
# GOOGLE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_tokens=3000,
    max_retries=2
)


# ============================================================
# GRAPH STATE
# ============================================================

class DevCrewState(TypedDict):

    user_request: str

    plan: str

    implementation: str

    review: str

    final_result: str


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get("text", "")

                    if text:
                        text_parts.append(text)

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# NODE 1: PLANNER
# ============================================================

def planner_node(state: DevCrewState):

    request = state["user_request"]

    prompt = f"""
You are the Planner in a software development team called Dev Crew.

The user wants to build the following:

{request}

Create a clear development plan.

Include:

1. Understanding of the requirement
2. Main features
3. Technologies that could be used
4. Project structure
5. Important implementation steps
6. Possible challenges

Do not write the complete code yet.

Keep the plan practical and suitable for a student software project.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "plan": extract_text(response)
    }


# ============================================================
# NODE 2: DEVELOPER
# ============================================================

def developer_node(state: DevCrewState):

    request = state["user_request"]

    plan = state["plan"]

    prompt = f"""
You are the Developer in a software development team called Dev Crew.

USER REQUIREMENT:

{request}

PLANNER'S PLAN:

{plan}

Now act as the main developer.

Based on the requirement and plan:

1. Explain the implementation approach.
2. Provide the important code or pseudocode needed.
3. Explain the main files and their purpose.
4. Mention how the components connect.
5. Make the solution practical and runnable.

Do not blindly follow the plan if you notice a better approach.

Use clear headings.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "implementation": extract_text(response)
    }


# ============================================================
# NODE 3: REVIEWER
# ============================================================

def reviewer_node(state: DevCrewState):

    request = state["user_request"]

    plan = state["plan"]

    implementation = state["implementation"]

    prompt = f"""
You are the Code Reviewer in a software development team called Dev Crew.

USER REQUIREMENT:

{request}

PLANNER'S PLAN:

{plan}

DEVELOPER'S IMPLEMENTATION:

{implementation}

Review the proposed solution carefully.

Check:

1. Whether the requirement is satisfied.
2. Technical correctness.
3. Missing features.
4. Possible bugs.
5. Security or reliability concerns.
6. Code quality.
7. Whether the approach is practical for a student project.

Give specific corrections and improvements.

If something is already correct, say so.

Do not rewrite the entire project.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "review": extract_text(response)
    }


# ============================================================
# NODE 4: FINALIZER
# ============================================================

def finalizer_node(state: DevCrewState):

    request = state["user_request"]

    plan = state["plan"]

    implementation = state["implementation"]

    review = state["review"]

    prompt = f"""
You are the Lead Developer of Dev Crew.

Create the final solution for the user's development request.

USER REQUIREMENT:

{request}

PLANNER:

{plan}

DEVELOPER:

{implementation}

REVIEWER:

{review}

Use the reviewer's feedback to improve the developer's solution.

Your final response must contain:

FINAL DEVELOPMENT SOLUTION

1. Requirement Understanding

2. Recommended Technologies

3. Project Structure

4. Implementation

5. Important Code

6. How It Works

7. Improvements Made After Review

8. Next Steps

Make the response practical and easy for a student to understand.

Do not mention internal agent instructions.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "final_result": extract_text(response)
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(DevCrewState)


# Add nodes

graph_builder.add_node(
    "planner",
    planner_node
)

graph_builder.add_node(
    "developer",
    developer_node
)

graph_builder.add_node(
    "reviewer",
    reviewer_node
)

graph_builder.add_node(
    "finalizer",
    finalizer_node
)


# Define graph flow

graph_builder.add_edge(
    START,
    "planner"
)

graph_builder.add_edge(
    "planner",
    "developer"
)

graph_builder.add_edge(
    "developer",
    "reviewer"
)

graph_builder.add_edge(
    "reviewer",
    "finalizer"
)

graph_builder.add_edge(
    "finalizer",
    END
)


# Compile graph

dev_crew = graph_builder.compile()


# ============================================================
# WEBSITE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Dev Crew</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
        }

        .container {
            max-width: 900px;
            margin: 30px auto;
            background: white;
            border-radius: 15px;
            padding: 25px;
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

        .description {
            text-align: center;
            color: #555;
            margin-bottom: 20px;
            line-height: 1.5;
        }

        textarea {
            width: 100%;
            min-height: 140px;
            resize: vertical;
            padding: 14px;
            border: 1px solid #ccc;
            border-radius: 10px;
            font-family: Arial, sans-serif;
            font-size: 16px;
        }

        button {
            display: block;
            margin: 15px auto;
            padding: 14px 25px;
            border: none;
            border-radius: 10px;
            background: #111827;
            color: white;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            opacity: 0.9;
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        #result {
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 20px;
            background: #fafafa;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 100px;
        }

        .status {
            text-align: center;
            color: #666;
            margin: 10px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Dev Crew</h1>

    <div class="subtitle">
        LangGraph Powered Development Team
    </div>

    <div class="description">
        Enter a software development requirement.
        Dev Crew will plan the solution, develop it,
        review it, and produce a final solution.
    </div>


    <textarea
        id="request"
        placeholder="Example: Build a student attendance management system using Python and MySQL."
    ></textarea>


    <button
        id="generateButton"
        onclick="generateSolution()"
    >
        Generate Solution
    </button>


    <div
        id="status"
        class="status"
    ></div>


    <div
        id="result"
    >
        Your final development solution will appear here.
    </div>

</div>


<script>

async function generateSolution() {

    const requestBox =
        document.getElementById("request");

    const button =
        document.getElementById("generateButton");

    const status =
        document.getElementById("status");

    const result =
        document.getElementById("result");


    const request =
        requestBox.value.trim();


    if (!request) {

        result.textContent =
            "Please enter a development requirement.";

        return;
    }


    button.disabled = true;

    status.textContent =
        "Dev Crew is working on your requirement...";

    result.textContent = "";


    try {

        const response = await fetch(
            "/generate",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    request: request
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            result.textContent =
                data.response ||
                "Something went wrong.";

        } else {

            result.textContent =
                data.response;
        }


    } catch (error) {

        result.textContent =
            "Unable to connect to the server. Please try again.";

    }


    status.textContent = "";

    button.disabled = false;
}


</script>

</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return HTML


# ============================================================
# GENERATE SOLUTION
# ============================================================

@app.post("/generate")
async def generate(request: Request):

    try:

        data = await request.json()

        user_request = data.get(
            "request",
            ""
        ).strip()


        if not user_request:

            return JSONResponse(
                {
                    "response":
                    "Please enter a development requirement."
                },
                status_code=400
            )


        initial_state = {

            "user_request": user_request,

            "plan": "",

            "implementation": "",

            "review": "",

            "final_result": ""
        }


        result = dev_crew.invoke(
            initial_state
        )


        final_result = result.get(
            "final_result",
            ""
        )


        return JSONResponse(
            {
                "response": final_result
            }
        )


    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            {
                "response":
                "Something went wrong while processing the request."
            },
            status_code=500
        )