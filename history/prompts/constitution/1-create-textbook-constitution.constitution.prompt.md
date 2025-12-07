---
id: 1
title: Create Textbook Constitution
stage: constitution
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, textbook, physical-ai, humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

```text
Project: Textbook on Physical AI & Humanoid Robotics for Hackathon

Core principles:
- **Comprehensiveness:** Cover key concepts in Physical AI and Humanoid Robotics.
- **Clarity:** Make content accessible for all learners, avoiding excessive jargon.
- **Reproducibility:** Ensure practical examples and algorithms are replicable.
- **Engagement:** Include real-world examples, case studies, and exercises.

Key standards:
- **Accuracy:** All claims verified through primary sources.
- **Citation format:** Use APA style.
- **Sources:** At least 40% peer-reviewed.
- **Plagiarism check:** 0% tolerance.
- **Clarity:** Target Flesch-Kincaid grade level 12-14.
- **Visual aids:** Include diagrams and tables.

Constraints:
- **Word count:** 40,000-50,000 words.
- **Sources:** Minimum 30 sources, with at least 40% peer-reviewed.
- **Format:** Markdown for Docusaurus, deployed to GitHub Pages.
- **Chapters:** 8-10 chapters with exercises.

Success criteria:
- **Verified claims** with proper citations.
- **Zero plagiarism** after checks.
- **Engaging, clear, and technically accurate** content.
- **Reviewed by experts** for accuracy.
```

## Response snapshot

Created a comprehensive constitution for the Physical AI & Humanoid Robotics textbook project. The constitution includes six core principles: Comprehensiveness, Clarity and Accessibility, Reproducibility and Verification, Evidence-Based Content, Engagement and Practical Application, and Visual Learning Support. It also establishes academic standards, development workflow, and governance requirements that align with the project's goals of creating an accessible, technically accurate, and engaging textbook for hackathon participants.

## Outcome

- ✅ Impact: Created foundational project constitution with comprehensive principles and standards
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md with project-specific principles
- 🔁 Next prompts: Begin creating spec for the textbook content
- 🧠 Reflection: Constitution successfully captures all core principles and requirements specified by the user

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A