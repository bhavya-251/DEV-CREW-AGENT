import uuid
from typing import TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# GOOGLE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_tokens=3500,
    max_retries=2
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict):
    user_request: str
    plan: str
    backend: str
    frontend: str
    review: str
    final_result: str


# ============================================================
# EXTRACT TEXT FROM GEMINI RESPONSE
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

    prompt = f"""
You are the PLANNER in a software development team called Dev Crew.

USER REQUIREMENT:

{state["user_request"]}

Analyze the requirement and create a practical project plan.

Include:

1. Requirement understanding
2. Main features
3. Recommended technologies
4. Database requirements if needed
5. Project structure
6. Main implementation steps
7. Important edge cases

Keep this response reasonably detailed but not excessively long.

Do NOT write source code.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "plan": extract_text(response)
    }


# ============================================================
# NODE 2: BACKEND DEVELOPER
# ============================================================

def backend_node(state: DevCrewState):

    prompt = f"""
You are the BACKEND DEVELOPER in the Dev Crew software
development team.

USER REQUIREMENT:

{state["user_request"]}

PROJECT PLAN:

{state["plan"]}

Now develop the backend portion of the requested project.

IMPORTANT:

- Provide complete backend implementation.
- Do not leave code unfinished.
- Do not use placeholders.
- Do not say "continue the code".
- Do not say "implementation omitted".
- Include important backend files.
- Include database models and logic when required.
- Make sure imports are correct.
- Make sure the code works with the planned project structure.
- Keep it practical for a student project.

Your response should contain:

1. Backend purpose
2. Backend files
3. Complete code for the backend files

Do NOT generate the frontend.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "backend": extract_text(response)
    }


# ============================================================
# NODE 3: FRONTEND DEVELOPER
# ============================================================

def frontend_node(state: DevCrewState):

    prompt = f"""
You are the FRONTEND DEVELOPER in the Dev Crew software
development team.

USER REQUIREMENT:

{state["user_request"]}

PROJECT PLAN:

{state["plan"]}

BACKEND IMPLEMENTATION:

{state["backend"]}

Now develop the frontend portion of the project.

IMPORTANT:

- Provide complete frontend implementation.
- Do not leave code unfinished.
- Do not use placeholders.
- Make the frontend compatible with the backend.
- Make sure routes and filenames match.
- Include HTML, CSS, and JavaScript when required.
- Do not rewrite the entire backend.

Your response should contain:

1. Frontend purpose
2. Frontend files
3. Complete frontend code

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "frontend": extract_text(response)
    }


# ============================================================
# NODE 4: REVIEWER
# ============================================================

def reviewer_node(state: DevCrewState):

    prompt = f"""
You are the REVIEWER in the Dev Crew software development team.

USER REQUIREMENT:

{state["user_request"]}

PROJECT PLAN:

{state["plan"]}

BACKEND:

{state["backend"]}

FRONTEND:

{state["frontend"]}

Review the proposed project carefully.

Check:

1. Requirement satisfaction
2. Backend correctness
3. Frontend correctness
4. Database consistency
5. Imports
6. Routes
7. File names
8. Dependencies
9. Obvious syntax problems
10. Obvious logical problems
11. Whether the project can realistically run

Give a concise review using these headings:

PROBLEMS FOUND

CORRECTIONS REQUIRED

WHAT IS ALREADY CORRECT

Do NOT rewrite the entire project.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "review": extract_text(response)
    }


# ============================================================
# NODE 5: FINALIZER
# ============================================================

def finalizer_node(state: DevCrewState):

    prompt = f"""
You are the FINALIZER and LEAD DEVELOPER of Dev Crew.

USER REQUIREMENT:

{state["user_request"]}

PROJECT PLAN:

{state["plan"]}

BACKEND:

{state["backend"]}

FRONTEND:

{state["frontend"]}

REVIEW:

{state["review"]}

Create the final project summary using the developer work
and the review.

IMPORTANT:

Do NOT regenerate the entire project code.

The backend and frontend code were already shown in the
previous stages.

Instead provide:

1. Final project structure
2. Technologies used
3. Corrections made after review
4. How the components work together
5. Installation steps
6. How to run the project
7. Important final notes

Keep this response reasonably concise.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "final_result": extract_text(response)
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)


builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "backend",
    backend_node
)

builder.add_node(
    "frontend",
    frontend_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.add_node(
    "finalizer",
    finalizer_node
)


# ============================================================
# GRAPH FLOW
# ============================================================

builder.add_edge(
    START,
    "planner"
)

builder.add_edge(
    "planner",
    "backend"
)

builder.add_edge(
    "backend",
    "frontend"
)

builder.add_edge(
    "frontend",
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


# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# COMPILE GRAPH
#
# The graph pauses AFTER each stage.
# Continue resumes the SAME thread.
# ============================================================

graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=[
        "planner",
        "backend",
        "frontend",
        "reviewer"
    ]
)


# ============================================================
# SESSION STORAGE
# ============================================================

sessions = {}


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

        #stage {
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
            color: #333;
        }

        #output {
            margin-top: 15px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 12px;
            background: #fafafa;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            min-height: 100px;
        }

        #continueButton {
            display: none;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>Dev Crew</h1>

    <div class="subtitle">
        LangGraph Powered Development Team
    </div>


    <textarea
        id="request"
        placeholder="Example: Build a student attendance management system using Python and MySQL."
    ></textarea>


    <button
        id="startButton"
        onclick="startProject()"
    >
        Start Project
    </button>


    <div id="stage"></div>


    <div id="output"></div>


    <button
        id="continueButton"
        onclick="continueProject()"
    >
        Continue
    </button>

</div>


<script>

let sessionId = null;


function showResult(data) {

    document.getElementById("stage").textContent =
        data.stage;


    document.getElementById("output").textContent =
        data.response;


    const button =
        document.getElementById("continueButton");


    if (data.finished) {

        button.style.display = "none";

        document.getElementById("startButton").disabled = false;

    } else {

        button.style.display = "block";

        button.disabled = false;
    }

}


async function startProject() {

    const request =
        document.getElementById("request").value.trim();


    if (!request) {

        document.getElementById("output").textContent =
            "Please enter a development requirement.";

        return;
    }


    document.getElementById("startButton").disabled = true;

    document.getElementById("continueButton").style.display =
        "none";

    document.getElementById("stage").textContent =
        "Planner is working...";

    document.getElementById("output").textContent = "";


    try {

        const response = await fetch(
            "/start",
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


        const data = await response.json();


        if (!response.ok) {

            document.getElementById("output").textContent =
                data.response || "Something went wrong.";

            document.getElementById("startButton").disabled =
                false;

            return;
        }


        sessionId = data.session_id;

        showResult(data);


    } catch (error) {

        document.getElementById("output").textContent =
            "Unable to connect to the server.";

        document.getElementById("startButton").disabled =
            false;
    }

}


async function continueProject() {

    const button =
        document.getElementById("continueButton");


    button.disabled = true;

    document.getElementById("stage").textContent =
        "Dev Crew is working...";


    try {

        const response = await fetch(
            "/continue",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    session_id: sessionId
                })
            }
        );


        const data = await response.json();


        if (!response.ok) {

            document.getElementById("output").textContent =
                data.response || "Something went wrong.";

            button.disabled = false;

            return;
        }


        showResult(data);


    } catch (error) {

        document.getElementById("output").textContent =
            "Unable to connect to the server.";

        button.disabled = false;
    }

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
# START PROJECT
# ============================================================

@app.post("/start")
async def start_project(request: Request):

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


        thread_id = str(uuid.uuid4())


        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }


        initial_state = {

            "user_request": user_request,

            "plan": "",

            "backend": "",

            "frontend": "",

            "review": "",

            "final_result": ""
        }


        result = graph.invoke(
            initial_state,
            config=config
        )


        sessions[thread_id] = True


        return JSONResponse(
            {
                "session_id": thread_id,
                "stage": "Planner",
                "response": result["plan"],
                "finished": False
            }
        )


    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            {
                "response":
                "Something went wrong while starting the project."
            },
            status_code=500
        )


# ============================================================
# CONTINUE PROJECT
# ============================================================

@app.post("/continue")
async def continue_project(request: Request):

    try:

        data = await request.json()

        thread_id = data.get(
            "session_id"
        )


        if not thread_id or thread_id not in sessions:

            return JSONResponse(
                {
                    "response":
                    "Project session not found. Please start again."
                },
                status_code=400
            )


        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }


        result = graph.invoke(
            None,
            config=config
        )


        state = result


        if state.get("final_result"):

            return JSONResponse(
                {
                    "stage": "Finalizer",
                    "response": state["final_result"],
                    "finished": True
                }
            )


        if state.get("review"):

            return JSONResponse(
                {
                    "stage": "Reviewer",
                    "response": state["review"],
                    "finished": False
                }
            )


        if state.get("frontend"):

            return JSONResponse(
                {
                    "stage": "Developer - Frontend",
                    "response": state["frontend"],
                    "finished": False
                }
            )


        if state.get("backend"):

            return JSONResponse(
                {
                    "stage": "Developer - Backend",
                    "response": state["backend"],
                    "finished": False
                }
            )


        return JSONResponse(
            {
                "stage": "Processing",
                "response":
                "Dev Crew is processing the project.",
                "finished": False
            }
        )


    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            {
                "response":
                "Something went wrong while continuing the project."
            },
            status_code=500
        )
