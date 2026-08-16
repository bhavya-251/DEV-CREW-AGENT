import io
import json
import re
import uuid
import zipfile
from typing import TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI


app = FastAPI()


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_output_tokens=16000,
    max_retries=2
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict):
    user_request: str
    plan: str
    backend_files: list
    frontend_files: list
    all_files: list
    current_index: int
    review: str
    final_result: str
    generated_files: list


# ============================================================
# HELPERS
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


def clean_code(code):
    code = code.strip()

    if code.startswith("```"):
        code = re.sub(
            r"^```[a-zA-Z0-9_+-]*\s*",
            "",
            code
        )

        code = re.sub(
            r"\s*```$",
            "",
            code
        )

    return code.strip()


def parse_json_response(text):

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "The model did not return valid JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


def normalize_file_list(items):

    result = []

    if not isinstance(items, list):
        return result

    for item in items:

        if not isinstance(item, dict):
            continue

        filename = str(
            item.get("filename", "")
        ).strip()

        purpose = str(
            item.get("purpose", "")
        ).strip()

        if filename:
            result.append({
                "filename": filename,
                "purpose": purpose
            })

    return result


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state):

    prompt = f"""
You are the PLANNER in a software development team.

USER REQUIREMENT:

{state["user_request"]}

Create a complete project plan and an exact file manifest.

Return ONLY valid JSON.

Do not use markdown.

Use exactly this structure:

{{
    "plan": "detailed project plan",
    "backend_files": [
        {{
            "filename": "app.py",
            "purpose": "main application"
        }}
    ],
    "frontend_files": [
        {{
            "filename": "templates/index.html",
            "purpose": "main frontend page"
        }}
    ]
}}

Rules:

- List every file that is actually needed.
- Do not list unnecessary files.
- Every listed file will later be generated completely.
- Keep the project suitable for a student.
- Make the backend and frontend compatible.
- Include requirements.txt if required.
- Do not include generated files such as __pycache__.
- Use real relative filenames.
- Do not use emojis.
"""

    response = llm.invoke(prompt)

    text = extract_text(response)

    try:

        data = parse_json_response(text)

        plan = str(
            data.get("plan", "")
        ).strip()

        backend_files = normalize_file_list(
            data.get("backend_files", [])
        )

        frontend_files = normalize_file_list(
            data.get("frontend_files", [])
        )

        all_files = (
            backend_files +
            frontend_files
        )

        if not all_files:
            raise ValueError(
                "No files returned."
            )

        return {
            "plan": plan,
            "backend_files": backend_files,
            "frontend_files": frontend_files,
            "all_files": all_files,
            "current_index": 0,
            "review": "",
            "final_result": "",
            "generated_files": []
        }

    except Exception:

        fallback = [
            {
                "filename": "app.py",
                "purpose":
                    "Main application"
            }
        ]

        return {
            "plan": text,
            "backend_files": fallback,
            "frontend_files": [],
            "all_files": fallback,
            "current_index": 0,
            "review": "",
            "final_result": "",
            "generated_files": []
        }


# ============================================================
# GENERATE ONE COMPLETE FILE
# ============================================================

def generate_file_node(state):

    all_files = state["all_files"]

    index = state["current_index"]

    if index >= len(all_files):
        return {}

    file_info = all_files[index]

    filename = file_info["filename"]

    purpose = file_info["purpose"]

    already_generated = [
        item["filename"]
        for item in state["generated_files"]
    ]

    previous_context = ""

    for item in state["generated_files"][-3:]:

        previous_context += (
            "\n\nFILE: "
            + item["filename"]
            + "\n"
            + item["content"]
        )

    prompt = f"""
You are a professional software developer.

USER REQUIREMENT:

{state["user_request"]}


PROJECT PLAN:

{state["plan"]}


CURRENT FILE:

{filename}


PURPOSE:

{purpose}


FILES ALREADY GENERATED:

{json.dumps(already_generated)}


PREVIOUS FILE CONTEXT:

{previous_context}


Generate ONLY the COMPLETE CONTENTS of:

{filename}


VERY IMPORTANT:

- Give the complete file.
- Do NOT give an explanation.
- Do NOT use markdown code fences.
- Do NOT use placeholders.
- Do NOT use "...".
- Do NOT use TODO.
- Do NOT leave functions incomplete.
- Do NOT say "same as above".
- Include every required import.
- Make the file compatible with the other files.
- If this is Python, give the complete Python file.
- If this is HTML, give the complete HTML.
- If this is CSS, give the complete CSS.
- If this is JavaScript, give the complete JavaScript.
- If this is requirements.txt, give every required package.
- Generate ONLY this one file.

The response MUST contain the COMPLETE file.
"""

    response = llm.invoke(prompt)

    content = clean_code(
        extract_text(response)
    )

    generated_files = list(
        state["generated_files"]
    )

    generated_files.append({
        "filename": filename,
        "content": content
    })

    return {
        "generated_files":
            generated_files,

        "current_index":
            index + 1
    }


# ============================================================
# DECIDE NEXT NODE
# ============================================================

def after_file_generation(state):

    if (
        state["current_index"]
        >= len(state["all_files"])
    ):
        return "reviewer"

    return "generate_file"


# ============================================================
# REVIEWER NODE
# ============================================================

def reviewer_node(state):

    project = ""

    for item in state["generated_files"]:

        project += (
            "\n\nFILE: "
            + item["filename"]
            + "\n"
            + item["content"]
        )

    prompt = f"""
You are the REVIEWER in a software development team.

USER REQUIREMENT:

{state["user_request"]}


PROJECT PLAN:

{state["plan"]}


GENERATED PROJECT:

{project}


Review the generated project.

Check:

1. Requirement satisfaction
2. Python syntax
3. Missing imports
4. Missing functions
5. Broken routes
6. Frontend/backend compatibility
7. Incorrect filenames
8. Missing dependencies
9. Incomplete code
10. Placeholder code
11. Whether the files work together

Return:

PROBLEMS FOUND

CORRECTIONS REQUIRED

OVERALL STATUS

Do NOT regenerate the project.

Do NOT repeat the complete source code.

Do not use emojis.
"""

    response = llm.invoke(prompt)

    return {
        "review":
            extract_text(response)
    }


# ============================================================
# FINALIZER NODE
# ============================================================

def finalizer_node(state):

    filenames = [
        item["filename"]
        for item in state["generated_files"]
    ]

    prompt = f"""
You are the FINALIZER of a software development team.

USER REQUIREMENT:

{state["user_request"]}


PROJECT PLAN:

{state["plan"]}


GENERATED FILES:

{json.dumps(filenames)}


REVIEW:

{state["review"]}


Prepare the final handoff.

Include:

1. Final project structure
2. What was generated
3. How to install dependencies
4. How to run the project
5. Important corrections
6. Any remaining issue

Do NOT regenerate source code.

Do not use emojis.
"""

    response = llm.invoke(prompt)

    return {
        "final_result":
            extract_text(response)
    }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(
    DevCrewState
)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "generate_file",
    generate_file_node
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
    "generate_file"
)

builder.add_conditional_edges(
    "generate_file",
    after_file_generation
)

builder.add_edge(
    "reviewer",
    "finalizer"
)

builder.add_edge(
    "finalizer",
    END
)


checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=[
        "generate_file",
        "reviewer"
    ]
)


# ============================================================
# SESSION STORAGE
# ============================================================

sessions = {}


# ============================================================
# FRONTEND
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
    max-width: 1000px;
    margin: 30px auto;
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow:
        0 5px 25px
        rgba(0, 0, 0, 0.08);
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
    padding: 12px 20px;
    border: none;
    border-radius: 9px;
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

.center-button {
    display: block;
    margin: 15px auto;
}

#stage {
    text-align: center;
    font-weight: bold;
    margin: 20px 0;
}

#output {
    margin-top: 15px;
}

.response-section {
    margin-bottom: 25px;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 12px;
    background: #fafafa;
}

.response-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 8px;
}

.response-file {
    color: #555;
    font-size: 14px;
    margin-bottom: 15px;
}

.response-content {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    overflow-x: auto;
    line-height: 1.5;
    font-family: Consolas, monospace;
    font-size: 13px;
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e2e2e2;
}

.download-button {
    margin-top: 15px;
    background: #374151;
}

#continueButton {
    display: none;
    margin: 20px auto;
}

#downloadZipButton {
    display: none;
    margin: 10px auto 20px auto;
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
    placeholder="Describe the application you want to build..."
></textarea>


<button
    id="startButton"
    class="center-button"
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


<button
    id="downloadZipButton"
    onclick="downloadZip()"
>
Download Complete Project ZIP
</button>


</div>


<script>

let sessionId = null;


function addResponse(data) {

    const output =
        document.getElementById(
            "output"
        );


    const section =
        document.createElement(
            "div"
        );

    section.className =
        "response-section";


    const title =
        document.createElement(
            "div"
        );

    title.className =
        "response-title";

    title.textContent =
        data.stage;

    section.appendChild(title);


    if (data.filename) {

        const fileName =
            document.createElement(
                "div"
            );

        fileName.className =
            "response-file";

        fileName.textContent =
            "File: " + data.filename;

        section.appendChild(
            fileName
        );
    }


    const content =
        document.createElement(
            "div"
        );

    content.className =
        "response-content";

    content.textContent =
        data.response;

    section.appendChild(
        content
    );


    if (
        data.filename &&
        data.response
    ) {

        const download =
            document.createElement(
                "button"
            );

        download.className =
            "download-button";

        download.textContent =
            "Download " +
            data.filename;


        download.onclick =
            function() {

                const blob =
                    new Blob(
                        [data.response],
                        {
                            type:
                            "text/plain;charset=utf-8"
                        }
                    );


                const url =
                    URL.createObjectURL(
                        blob
                    );


                const a =
                    document.createElement(
                        "a"
                    );


                a.href = url;


                a.download =
                    data.filename
                    .split("/")
                    .pop();


                document.body.appendChild(
                    a
                );


                a.click();


                a.remove();


                URL.revokeObjectURL(
                    url
                );
            };


        section.appendChild(
            download
        );
    }


    output.appendChild(
        section
    );


    window.scrollTo({
        top:
            document.body.scrollHeight,
        behavior:
            "smooth"
    });


    const continueButton =
        document.getElementById(
            "continueButton"
        );


    if (data.finished) {

        continueButton.style.display =
            "none";


        document.getElementById(
            "downloadZipButton"
        ).style.display =
            "block";


        document.getElementById(
            "startButton"
        ).disabled = false;


        document.getElementById(
            "stage"
        ).textContent =
            "Project completed.";

    } else {

        continueButton.style.display =
            "block";

        continueButton.disabled =
            false;

        document.getElementById(
            "stage"
        ).textContent = "";
    }
}


async function startProject() {

    const request =
        document.getElementById(
            "request"
        ).value.trim();


    if (!request) {

        alert(
            "Please enter a requirement."
        );

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
        "downloadZipButton"
    ).style.display =
        "none";


    document.getElementById(
        "output"
    ).innerHTML = "";


    document.getElementById(
        "stage"
    ).textContent =
        "Creating project plan...";


    try {

        const response =
            await fetch(
                "/start",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            request:
                                request
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.response
            );
        }


        sessionId =
            data.session_id;


        addResponse(data);

    }

    catch (error) {

        document.getElementById(
            "output"
        ).textContent =
            error.message;


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
        "Generating next file...";


    try {

        const response =
            await fetch(
                "/continue",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            session_id:
                                sessionId
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.response
            );
        }


        addResponse(data);

    }

    catch (error) {

        document.getElementById(
            "stage"
        ).textContent =
            error.message;

        button.disabled = false;
    }
}


function downloadZip() {

    if (!sessionId) {
        return;
    }


    window.location.href =
        "/download/" + sessionId;
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
# START PROJECT
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
                        "Please enter a requirement."
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

            "backend_files":
                [],

            "frontend_files":
                [],

            "all_files":
                [],

            "current_index":
                0,

            "review":
                "",

            "final_result":
                "",

            "generated_files":
                []
        }


        result =
            graph.invoke(
                initial_state,
                config=config
            )


        sessions[
            thread_id
        ] = True


        return JSONResponse({

            "session_id":
                thread_id,

            "stage":
                "Planner",

            "response":
                result.get(
                    "plan",
                    ""
                ),

            "finished":
                False
        })


    except Exception as exc:

        print(
            "START ERROR:",
            repr(exc)
        )


        return JSONResponse(
            {
                "response":
                    "Something went wrong: "
                    + str(exc)
            },
            status_code=500
        )


# ============================================================
# CONTINUE PROJECT
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
                        "Session not found. Start again."
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


        generated_files =
            result.get(
                "generated_files",
                []
            )


        all_files =
            result.get(
                "all_files",
                []
            )


        current_index =
            result.get(
                "current_index",
                0
            )


        # ----------------------------------------------------
        # A file was just generated
        # ----------------------------------------------------

        if (
            generated_files
            and current_index <= len(all_files)
            and current_index > 0
        ):

            last_file =
                generated_files[-1]


            if current_index <= len(
                all_files
            ):

                return JSONResponse({

                    "stage":
                        "Developer",

                    "filename":
                        last_file[
                            "filename"
                        ],

                    "response":
                        last_file[
                            "content"
                        ],

                    "finished":
                        False
                })


        # ----------------------------------------------------
        # Reviewer
        # ----------------------------------------------------

        if result.get("review"):

            return JSONResponse({

                "stage":
                    "Reviewer",

                "response":
                    result["review"],

                "finished":
                    False
            })


        # ----------------------------------------------------
        # Finalizer
        # ----------------------------------------------------

        if result.get(
            "final_result"
        ):

            return JSONResponse({

                "stage":
                    "Finalizer",

                "response":
                    result[
                        "final_result"
                    ],

                "finished":
                    True
            })


        return JSONResponse({

            "stage":
                "Processing",

            "response":
                "Processing the project...",

            "finished":
                False
        })


    except Exception as exc:

        print(
            "CONTINUE ERROR:",
            repr(exc)
        )


        return JSONResponse(
            {
                "response":
                    "Something went wrong: "
                    + str(exc)
            },
            status_code=500
        )


# ============================================================
# DOWNLOAD COMPLETE PROJECT
# ============================================================

@app.get(
    "/download/{session_id}"
)
async def download_project(
    session_id: str
):

    if session_id not in sessions:

        return JSONResponse(
            {
                "response":
                    "Session not found."
            },
            status_code=404
        )


    config = {
        "configurable": {
            "thread_id":
                session_id
        }
    }


    state =
        graph.get_state(
            config
        )


    generated_files =
        state.values.get(
            "generated_files",
            []
        )


    if not generated_files:

        return JSONResponse(
            {
                "response":
                    "No files generated."
            },
            status_code=404
        )


    memory_file =
        io.BytesIO()


    with zipfile.ZipFile(
        memory_file,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for item in generated_files:

            filename =
                item["filename"].replace(
                    "\\",
                    "/"
                )


            zip_file.writestr(
                filename,
                item["content"]
            )


    memory_file.seek(0)


    return StreamingResponse(

        memory_file,

        media_type=
            "application/zip",

        headers={
            "Content-Disposition":
                'attachment; filename="dev_crew_project.zip"'
        }
    )
