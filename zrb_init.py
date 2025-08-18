import os

from zrb import CmdTask, Group, cli

get_cwd = lambda ctx: os.path.dirname(__file__)

calo_group = cli.add_group(Group("calo", description="Calo Management"))


create_venv = CmdTask(
    name="create-calo-venv",
    cwd=get_cwd,
    cmd=["python -m venv .venv", "source .venv/bin/activate", "poetry install"],
)

publish = calo_group.add_task(
    CmdTask(
        name="publish-calo",
        cwd=get_cwd,
        cmd=[
            "source .venv/bin/activate",
            "./publish.sh",
        ],
    ),
    alias="publish",
)
create_venv >> publish

test = calo_group.add_task(
    CmdTask(
        name="test-calo",
        cwd=get_cwd,
        cmd=[
            "source .venv/bin/activate",
            "./test.sh",
        ],
    ),
    alias="test",
)
create_venv >> test

start = calo_group.add_task(
    CmdTask(
        name="start-calo",
        cwd=get_cwd,
        cmd=[
            "source .venv/bin/activate",
            "if [ -f .env ]",
            "then",
            "  source .env",
            "fi",
            "gemini-calo",
        ],
    ),
    alias="start",
)
create_venv >> start
