import os
from typing import TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

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
    max_tokens=5000,
    max_retries=2
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict):
    user_request: str
    plan: str
    implementation: str
    review: str
    final_result: str


# ============================================================
# RESPONSE TEXT HELPER
# ============================================================

def extract_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text = item.get("text", "")

                    if text:
                        parts.append(text)

            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts)

    return str(content)


# ============================================================
# NODE 1: PLANNER
# ============================================================

def planner_node(state: DevCrewState):

    request = state["user_request"]

    prompt = f"""
You are the PLANNER in a software development team called Dev Crew.

The user has requested:

{request}

Analyze the requirement and create a practical development plan.

Your plan must contain:

1. What the application should do
2. Main features
3. Recommended technology stack
4. Database requirements if needed
5. Main files/components required
6. Important implementation steps
7. Potential problems or edge cases

Keep the plan focused and practical.

Do NOT write the actual source code.

Do NOT generate a long essay.

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
You are the DEVELOPER in the Dev Crew software development team.

USER REQUIREMENT:
{request}

PLANNER'S PLAN:
{plan}

Now implement the requested application.

IMPORTANT:

The goal is to produce an ACTUAL WORKING STUDENT PROJECT.

You must:

1. Follow the planner's useful recommendations.
2. Choose a simple and practical implementation.
3. Provide COMPLETE code for the important files.
4. Do not intentionally leave code unfinished.
5. Do not write placeholders such as:
   - "add your code here"
   - "continue the code"
   - "etc."
   - "implementation omitted"
6. Make sure the files work together.
7. Include requirements.txt when external Python packages are needed.
8. Include database setup when a database is required.
9. Include HTML/templates when a web application requires them.
10. Keep the implementation realistic for a student project.

Start with a short implementation summary.

Then provide the project structure.

Then provide the complete code.

Do not spend most of the response explaining theory.

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
    implementation = state["implementation"]

    prompt = f"""
You are the REVIEWER in the Dev Crew software development team.

USER REQUIREMENT:
{request}

DEVELOPER'S IMPLEMENTATION:
{implementation}

Review the developer's solution.

Check specifically:

1. Does it satisfy the user's requirement?
2. Is the code complete?
3. Are there syntax or logical problems?
4. Are required dependencies included?
5. Do file names and imports match?
6. Are database operations consistent?
7. Are routes, functions, and templates connected correctly?
8. Are there obvious security problems?
9. Can a student realistically run the project?

Give a concise review.

List:
- Problems found
- Required corrections
- Things that are already correct

Do NOT rewrite the entire project.

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
You are the FINALIZER and LEAD DEVELOPER of Dev Crew.

Your job is to turn the developer's implementation into the
FINAL, CLEAN, USABLE answer.

USER REQUIREMENT:
{request}

PLANNER:
{plan}

DEVELOPER:
{implementation}

REVIEWER:
{review}

IMPORTANT FINAL OUTPUT RULES:

The final answer must prioritize a COMPLETE WORKING SOLUTION.

Use this structure:

# FINAL DEVELOPMENT SOLUTION

## 1. Technology Stack

Give a short list of technologies.

## 2. Project Structure

Show the required files and folders.

## 3. Requirements

Show the complete requirements.txt if needed.

## 4. Implementation

Provide COMPLETE code for the important files.

VERY IMPORTANT:

- Do not stop halfway through a file.
- Do not use placeholders.
- Do not say "remaining code is similar".
- Do not say "continue here".
- Do not omit important code.
- Make sure imports match the project structure.
- Make sure functions and routes referenced by other files actually exist.
- Apply the reviewer's corrections.
- Keep the implementation practical.
- Prefer a smaller COMPLETE project over a huge INCOMPLETE project.

## 5. How to Run

Give simple commands to install and run the project.

## 6. How It Works

Give a short explanation of the main workflow.

Do NOT write a long theoretical explanation.

Do NOT include internal planner/developer/reviewer discussions.

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


# ============================================================
# GRAPH FLOW
# ============================================================

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


# ============================================================
# COMPILE GRAPH
# ============================================================

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


    <div id="result">
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


        result.textContent =
            data.response ||
            "No response was generated.";


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
# GENERATE SOLUTION ROUTE
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


        if not final_result:

            final_result = (
                "The development team could not "
                "generate a final solution."
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
