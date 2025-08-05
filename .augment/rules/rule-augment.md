---
type: "always_apply"
---

These rules should be applied after every code generation or modification prompt to ensure consistency, quality, and maintainability across the codebase.

## 📁 File Management & Naming Conventions

1. **Temporary Files:**
   - For any temporary summary, scratchpad, or local reference files (e.g., Markdown notes, temporary code snippets), use naming conventions that allow for easy exclusion from version control.
   - **Recommended Naming:** `*augment.md`, `*local.md`, `*temp.txt`, or similar patterns.
   - **Action:** Ensure these patterns are added to your project's `.gitignore` file to prevent accidental commits.

## 🧹 Code Quality & Linting

1. **Automated Linting & Formatting:**
   - After _any_ file is updated (including new files, modified existing files, and even test files), run the specified linting and formatting tools.
   - **Mandatory Checks:**
     - **Ruff:** For general Python linting and formatting.
     - **Mypy:** For static type checking to ensure type consistency.
     - **Bandit:** For identifying common security issues in Python code.
   - **Action:** Address all reported warnings and errors from these tools. The goal is to have a clean linting report for all changed files.
2. **Readability & Clarity:**
   - **Self-Documenting Code:** Write code that is as clear and understandable as possible without excessive comments. Use meaningful variable, function, and class names.
   - **Consistent Style:** Adhere to the project's established coding style (e.g., PEP 8 for Python).
   - **Modularity:** Break down complex functions or classes into smaller, single-responsibility units.
3. **Comments & Documentation:**
   - **Purpose:** Use comments to explain _why_ certain decisions were made, complex algorithms, or non-obvious logic, rather than just _what_ the code does (which should be evident from self-documenting code).
   - **Function/Class Docstrings:** All public functions, methods, and classes should have clear docstrings explaining their purpose, arguments, return values, and any exceptions they might raise.

## 🧪 Testing & Reliability

1. **Test Coverage:**
   - For every new feature or bug fix, ensure corresponding unit and, where appropriate, integration tests are written or updated.
   - **Action:** Run the full test suite after changes to confirm no regressions have been introduced.
2. **Edge Cases & Error Handling:**
   - **Robustness:** Consider potential edge cases, invalid inputs, and error conditions.
   - **Graceful Degradation:** Implement appropriate error handling (e.g., `try-except` blocks, validation) to prevent crashes and provide informative feedback.

## 🔒 Security Considerations

1. **Input Validation:**
   - **Trust No Input:** Always validate and sanitize all user inputs and external data to prevent common vulnerabilities like injection attacks (SQL, command, XSS).
2. **Dependency Security:**
   - **Vulnerability Scanning:** Be mindful of the security implications of third-party libraries. Regularly check for known vulnerabilities in dependencies.
3. **Least Privilege:**
   - Ensure that code and processes operate with the minimum necessary permissions.

## ⚡ Performance Optimizations

1. **Efficiency:**
   - **Algorithm Choice:** Select efficient algorithms and data structures for the task at hand.
   - **Resource Management:** Properly manage resources (e.g., file handles, database connections) to prevent leaks.
   - **Avoid Redundancy:** Refactor duplicate code and avoid unnecessary computations.

## 📖 Documentation

1. **README Updates:**
   - If the changes introduce new features, configurations, or significant architectural shifts, update the project's `README.md` file.
2. **API Documentation:**
   - Ensure any public APIs or interfaces are well-documented.

## 🔄 Development Workflow (General)

1. **Atomic Commits:**
   - Keep changes small and focused on a single logical unit of work. This makes code reviews easier and simplifies debugging.
2. **Review Before Finalization:**
   - Before considering the augmentation complete, perform a self-review of the changes against these guidelines.

By consistently applying these rules, we can ensure a high standard of code quality, security, and maintainability across all projects.
