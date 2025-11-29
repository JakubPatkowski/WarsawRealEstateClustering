from invoke import Context, task

TARGET = "."


@task
def sort_imports(c: Context) -> None:
    """Sort imports using isort."""
    print("🔹 Running isort...")
    c.run(f"isort {TARGET}")


@task
def format_code(c: Context) -> None:
    """Format code using black."""
    print("🔹 Running black...")
    c.run(f"black {TARGET}")


@task
def lint(c: Context) -> None:
    """Lint code using ruff."""
    print("🔹 Running ruff...")
    c.run(f"ruff check {TARGET} --fix")


@task
def type_check(c: Context) -> None:
    """Check types using mypy."""
    print("🔹 Running mypy...")
    c.run(f"mypy {TARGET}")


@task
def security_check(c: Context) -> None:
    """Check for security issues using bandit."""
    print("🔹 Running bandit...")
    # -r is recursive, -q is quiet (optional), -ll is severity level (optional)
    c.run(f"bandit {TARGET}")


@task(pre=[sort_imports, format_code, lint, type_check, security_check])
def all(c: Context) -> None:
    """Run all formatters, linters, and checks in order."""
    print("\n✅ All checks completed successfully!")
