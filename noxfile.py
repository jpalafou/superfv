import nox


@nox.session
def test(session):
    session.run("pytest", external=True)


@nox.session
def lint(session):
    session.run("pre-commit", "run", "--all-files", external=True)


@nox.session
def typecheck(session):
    session.run("mypy", "superfv", "--check-untyped-defs", external=True)
