import asyncio
import os
from typing import Dict, Any, List

import dotenv
from github import Github, Auth
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

github_token = os.getenv("GITHUB_TOKEN")
auth = Auth.Token(github_token) if github_token else None
github_client = Github(auth=auth) if os.getenv("GITHUB_TOKEN") else Github()

repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

if github_client is not None:
    repo = github_client.get_repo(repository)


def get_pr_details(number: int) -> Dict[str, Any]:
    """
    Fetches the details of a pull request using its number.

    This function retrieves detailed information about a pull request from a repository.
    Details include the pull request's author, title, body content, URL of the diff,
    state, and a list of commit SHAs.

    :param number: The number of the pull request to retrieve details for.
    :type number: int
    :return: A dictionary containing author, title, body, diff URL, state, and
             commit SHAs of the pull request.
    :rtype: Dict[str, Any]
    """
    pull_request = repo.get_pull(number=number)
    commit_SHAs = []
    commits = pull_request.get_commits()

    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body if pull_request.body else "<empty>",
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commits": commit_SHAs,
    }


def get_file_content(file_path: str) -> str:
    """
    Retrieve and decode the content of a specified file.

    This function accesses the repository to fetch the desired file from
    the branch and decodes its content into a readable string format. It
    is primarily used to handle file operations and retrieve textual
    files from the repository for further processing.

    :param file_path: The path of the file to be retrieved.
    :return: A string containing the decoded content of the file.
    :rtype: str
    """

    return repo.get_contents(file_path).decoded_content.decode('utf-8')


def get_commit_details(commit_sha: str) -> List[Dict[str, Any]]:
    """
    Retrieve details of the files changed in a specific commit.

    This function fetches details about all the files that were modified in the commit
    identified by the given commit SHA. It retrieves information such as filename,
    status (added, modified, or removed), additions, deletions, total changes,
    and the patch data for each file.

    :param commit_sha: The SHA string of the commit to extract details from.
    :type commit_sha: str
    :return: A list of dictionaries where each dictionary contains details about a
             file changed in the specified commit.
    :rtype: List[Dict[str, Any]]
    """
    commit = repo.get_commit(commit_sha)
    changed_files: List[Dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })

    return changed_files


async def add_comment_to_state(ctx: Context, review_comment: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["review_comment"] = review_comment
    await ctx.store.set("state", current_state)
    return f"Review comment updated in state to: {review_comment}"


async def add_context_to_state(ctx: Context, context: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] += f"\n{context}"
    await ctx.store.set("state", current_state)
    return f"Context updated in state to: {context}"


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["final_review_comment"] = final_review
    await ctx.store.set("state", current_state)
    return f"Final review updated in state to: {final_review}"


def post_review_to_github(pr_number: int, review_body: str) -> None:
    """
    Posts a review to a specified pull request on GitHub.

    This function interacts with the GitHub repository to post a review
    to a given pull request. The review consists of a body of text
    containing the feedback or comments. It integrates with GitHub's
    API to achieve this functionality.

    :param pr_number: The number of the pull request to which the review will be posted.
    :type pr_number: int
    :param review_body: The content of the review to be submitted.
    :type review_body: str
    :return: None
    """
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=review_body)


get_pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
)
get_file_content_tool = FunctionTool.from_defaults(
    get_file_content,
)
get_commit_details_tool = FunctionTool.from_defaults(
    get_commit_details,
)
post_review_to_github_tool = FunctionTool.from_defaults(
    post_review_to_github,
)

context_agent_system_prompt = """You are the context gathering agent. When gathering context, you MUST gather 
    - The details: author, title, body, diff_url, state, and head_sha;
    - Changed files;
    - Any requested for files;
    Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers context for the pull request review comment.",
    tools=[get_pr_details_tool, get_commit_details_tool, get_file_content_tool, add_context_to_state],
    system_prompt=context_agent_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

commentator_agent_system_prompt = """You are the commentator agent that writes review comments for pull requests as a human reviewer would. 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the Commentor Agent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
 """

commentator_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent_system_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub."""
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Posts the review comment to GitHub after final checks.",
    tools=[add_final_review_to_state, post_review_to_github_tool],
    system_prompt=review_and_posting_agent_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentator_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review_comment": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    github_client.close()
