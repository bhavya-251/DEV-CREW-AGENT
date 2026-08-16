import uuid
from typing import TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI


app = FastAPI()


# ============================================================
# GOOGLE MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_tokens=8000,
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
# RESPONSE TEXT EXTRACTION
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
# PLANNER AGENT
# ============================================================

def planner_node(state: DevCrewState):

    prompt = f"""
You are the PLANNER of a software development team called Dev Crew.

The user has requested:

{state["user_request"]}

Create a clear and practical development plan.

Include:

1. Requirement understanding
2. Main features
3. User roles if applicable
4. Recommended technologies
5. Database requirements if applicable
6. Complete project folder structure
7. Backend requirements
8. Frontend requirements
9. Important implementation steps
10. Important edge cases
11. Required dependencies

This is the planning stage, so do not generate the complete source
code yet.

Keep the plan detailed enough for the developers to implement it.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "plan": extract_text(response)
    }


# ============================================================
# BACKEND DEVELOPER AGENT
# ============================================================

def backend_node(state: DevCrewState):

    prompt = f"""
You are the BACKEND DEVELOPER in a professional software
development team called Dev Crew.

USER REQUIREMENT:
{state["user_request"]}

PROJECT PLAN:
{state["plan"]}

Your job is to implement the COMPLETE BACKEND of this project.

VERY IMPORTANT:

You MUST provide the FULL SOURCE CODE.

Do NOT provide only an explanation.

Do NOT provide pseudocode.

Do NOT provide partial code.

Do NOT use placeholders such as:

...
pass
TODO
"add your code here"
"implement this"
"rest of the code"
"same as above"
"code omitted"

Every required backend file must be shown completely.

For example, if the backend requires:

app.py
models.py
routes.py
database.py

then provide the COMPLETE contents of every one of those files.

For every file, clearly show its filename and then its complete
code in a code block.

Make sure:

- All imports are included.
- All functions are included.
- All classes are included.
- Database models are complete.
- API routes are complete.
- Authentication is complete if required.
- Validation is complete if required.
- Error handling is included where appropriate.
- Files work together.
- Imports between files match the folder structure.
- The code is runnable.
- Do not invent files that are unnecessary.
- Include requirements.txt if backend dependencies are required.

If the project can reasonably be implemented as one backend file,
give the complete single file.

Do not generate the frontend in this stage.

Do not use emojis or decorative symbols.

Your response MUST contain:

1. Backend overview
2. Backend project structure
3. Complete backend source code for EVERY required file
4. Backend dependencies
"""


    response = llm.invoke(prompt)

    return {
        "backend": extract_text(response)
    }


# ============================================================
# FRONTEND DEVELOPER AGENT
# ============================================================

def frontend_node(state: DevCrewState):

    prompt = f"""
You are the FRONTEND DEVELOPER in a professional software
development team called Dev Crew.

USER REQUIREMENT:
{state["user_request"]}

PROJECT PLAN:
{state["plan"]}

BACKEND IMPLEMENTATION:
{state["backend"]}

Your job is to implement the COMPLETE FRONTEND of this project.

VERY IMPORTANT:

You MUST provide the FULL SOURCE CODE.

Do NOT provide only an explanation.

Do NOT provide pseudocode.

Do NOT provide partial code.

Do NOT use placeholders such as:

...
pass
TODO
"add your code here"
"implement this"
"rest of the code"
"same as above"
"code omitted"

Every required frontend file must be shown completely.

For example, if the frontend requires:

templates/base.html
templates/index.html
templates/login.html
static/css/style.css
static/js/script.js

then provide the COMPLETE contents of every one of those files.

For every file:

1. Clearly show the filename.
2. Give the COMPLETE code.
3. Make sure the code is directly usable.

Make sure:

- HTML is complete.
- CSS is complete.
- JavaScript is complete if required.
- Forms are complete.
- Navigation is complete.
- API calls match the backend.
- URLs/routes match the backend.
- Template names match the backend.
- IDs and class names match the JavaScript and CSS.
- No frontend functionality is left unfinished.
- The UI is clean and usable.
- The frontend actually works with the provided backend.

If the project does not need JavaScript, do not invent unnecessary
JavaScript.

Do not rewrite the backend unless a frontend/backend connection
requires a small correction. If such a correction is necessary,
clearly mention it.

Do not use emojis or decorative symbols.

Your response MUST contain:

1. Frontend overview
2. Frontend project structure
3. Complete frontend source code for EVERY required file
4. Any frontend dependencies if required
"""


    response = llm.invoke(prompt)

    return {
        "frontend": extract_text(response)
    }


# ============================================================
# REVIEWER AGENT
# ============================================================

def reviewer_node(state: DevCrewState):

    prompt = f"""
You are the REVIEWER of the Dev Crew software development team.

USER REQUIREMENT:
{state["user_request"]}

PROJECT PLAN:
{state["plan"]}

BACKEND IMPLEMENTATION:
{state["backend"]}

FRONTEND IMPLEMENTATION:
{state["frontend"]}

Review the complete project.

Check carefully:

1. Does the implementation satisfy the user requirement?
2. Is the backend complete?
3. Is the frontend complete?
4. Do backend routes match frontend requests?
5. Do frontend URLs match backend routes?
6. Are imports correct?
7. Are filenames correct?
8. Are dependencies correct?
9. Are database models and relationships correct?
10. Are there obvious syntax errors?
11. Are there obvious logical errors?
12. Are there missing files?
13. Are there missing functions?
14. Are there placeholder sections?
15. Can the project realistically run?

Use these headings:

PROBLEMS FOUND

CORRECTIONS REQUIRED

BACKEND CHECK

FRONTEND CHECK

WHAT IS ALREADY CORRECT

IMPORTANT:

Do NOT regenerate the entire project.

Do NOT repeat all source code.

Focus on identifying problems and exactly what should be corrected.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "review": extract_text(response)
    }


# ============================================================
# FINALIZER AGENT
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

Prepare the final project handoff.

The Backend Developer and Frontend Developer already provided
complete source code.

Do NOT unnecessarily regenerate all source code.

Instead provide:

1. Final project structure
2. Technologies used
3. Required dependencies
4. Corrections that should be applied based on the review
5. How the backend and frontend connect
6. Installation steps
7. How to run the project
8. Important final notes
9. Any remaining issue that the developer should know about

If the reviewer found a serious problem, explain the exact
correction needed.

Do not use emojis or decorative symbols.
"""

    response = llm.invoke(prompt)

    return {
        "final_result": extract_text(response)
    }


# ============================================================
# CREATE LANGGRAPH
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
# GRAPH CONNECTIONS
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
# CHECKPOINT
# ============================================================

checkpointer = InMemorySaver()


graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=[
        "planner",
        "backend",
        "frontend",
        "reviewer"
    ]
)


sessions = {}


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>Dev Crew</title>

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
            background: #f5f7fb;
        }

        .container {
            max-width: 950px;
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
            min-height: 150px;
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
            line-height: 1.6;
            min-height: 100px;
        }

        .response-section {
            padding-bottom: 30px;
            margin-bottom: 30px;
            border-bottom: 1px solid #ddd;
        }

        .response-section:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        .response-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        .response-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: anywhere;
        }

        .code-block {
            overflow-x: auto;
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


function addResponse(data) {

    const output =
        document.getElementById("output");


    const section =
        document.createElement("div");

    section.className =
        "response-section";


    const title =
        document.createElement("div");

    title.className =
        "response-title";

    title.textContent =
        data.stage;


    const content =
        document.createElement("div");

    content.className =
        "response-content";

    content.textContent =
        data.response;


    section.appendChild(title);

    section.appendChild(content);

    output.appendChild(section);


    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });


    const button =
        document.getElementById(
            "continueButton"
        );


    if (data.finished) {

        button.style.display =
            "none";

        document.getElementById(
            "startButton"
        ).disabled = false;

    } else {

        button.style.display =
            "block";

        button.disabled = false;
    }

}


async function startProject() {

    const request =
        document.getElementById(
            "request"
        ).value.trim();


    if (!request) {

        document.getElementById(
            "output"
        ).textContent =
            "Please enter a development requirement.";

        return;
    }


    document.getElementById(
        "startButton"
    ).disabled = true;


    document.getElementById(
        "continueButton"
    ).style.display =
        "none";


    document.getElementById(
        "stage"
    ).textContent =
        "Planner is working...";


    document.getElementById(
        "output"
    ).innerHTML = "";


    try {

        const response =
            await fetch(
                "/start",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        request:
                            request
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            document.getElementById(
                "output"
            ).textContent =
                data.response ||
                "Something went wrong.";

            document.getElementById(
                "startButton"
            ).disabled = false;

            return;
        }


        sessionId =
            data.session_id;


        addResponse(data);


    } catch (error) {

        document.getElementById(
            "output"
        ).textContent =
            "Unable to connect to the server.";

        document.getElementById(
            "startButton"
        ).disabled = false;
    }

}


async function continueProject() {

    const button =
        document.getElementById(
            "continueButton"
        );


    button.disabled = true;


    document.getElementById(
        "stage"
    ).textContent =
        "Dev Crew is working...";


    try {

        const response =
            await fetch(
                "/continue",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        session_id:
                            sessionId
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            document.getElementById(
                "output"
            ).textContent =
                data.response ||
                "Something went wrong.";

            button.disabled = false;

            return;
        }


        addResponse(data);


    } catch (error) {

        document.getElementById(
            "output"
        ).textContent =
            "Unable to connect to the server.";

        button.disabled = false;
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
# START
# ============================================================

@app.post("/start")
async def start_project(
    request: Request
):

    try:

        data =
            await request.json()

        user_request =
            data.get(
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


        thread_id =
            str(uuid.uuid4())


        config = {
            "configurable": {
                "thread_id":
                    thread_id
            }
        }


        initial_state = {

            "user_request":
                user_request,

            "plan":
                "",

            "backend":
                "",

            "frontend":
                "",

            "review":
                "",

            "final_result":
                ""
        }


        result =
            graph.invoke(
                initial_state,
                config=config
            )


        sessions[
            thread_id
        ] = True


        return JSONResponse(
            {
                "session_id":
                    thread_id,

                "stage":
                    "Planner",

                "response":
                    result["plan"],

                "finished":
                    False
            }
        )


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        return JSONResponse(
            {
                "response":
                "Something went wrong while starting the project."
            },
            status_code=500
        )


# ============================================================
# CONTINUE
# ============================================================

@app.post("/continue")
async def continue_project(
    request: Request
):

    try:

        data =
            await request.json()


        thread_id =
            data.get(
                "session_id"
            )


        if (
            not thread_id
            or thread_id not in sessions
        ):

            return JSONResponse(
                {
                    "response":
                    "Project session not found. Please start again."
                },
                status_code=400
            )


        config = {
            "configurable": {
                "thread_id":
                    thread_id
            }
        }


        result =
            graph.invoke(
                None,
                config=config
            )


        state =
            result


        if state.get(
            "final_result"
        ):

            return JSONResponse(
                {
                    "stage":
                        "Finalizer",

                    "response":
                        state[
                            "final_result"
                        ],

                    "finished":
                        True
                }
            )


        if state.get(
            "review"
        ):

            return JSONResponse(
                {
                    "stage":
                        "Reviewer",

                    "response":
                        state[
                            "review"
                        ],

                    "finished":
                        False
                }
            )


        if state.get(
            "frontend"
        ):

            return JSONResponse(
                {
                    "stage":
                        "Developer - Frontend",

                    "response":
                        state[
                            "frontend"
                        ],

                    "finished":
                        False
                }
            )


        if state.get(
            "backend"
        ):

            return JSONResponse(
                {
                    "stage":
                        "Developer - Backend",

                    "response":
                        state[
                            "backend"
                        ],

                    "finished":
                        False
                }
            )


        return JSONResponse(
            {
                "stage":
                    "Processing",

                "response":
                    "Dev Crew is processing the project.",

                "finished":
                    False
            }
        )


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        return JSONResponse(
            {
                "response":
                "Something went wrong while continuing the project."
            },
            status_code=500
        )
