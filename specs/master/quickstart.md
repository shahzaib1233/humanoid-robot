# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

Before contributing to the textbook, ensure you have the following installed:

- **Node.js**: Version 18 or higher
- **Git**: For version control
- **A code editor**: VS Code, Vim, or your preferred editor
- **Markdown knowledge**: Basic understanding of Markdown syntax

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/humainoid-robots-book.git
cd humainoid-robots-book
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Install Docusaurus

```bash
npm install @docusaurus/core@latest @docusaurus/preset-classic@latest
```

### 4. Start Development Server

```bash
npm start
```

This will start a local development server at `http://localhost:3000` where you can preview your changes in real-time.

## Project Structure

```
humainoid-robots-book/
├── docs/                    # Textbook content
│   ├── intro/              # Introduction chapter
│   ├── theoretical-foundations/ # Theory chapters
│   ├── practical-implementation/ # Practical chapters
│   ├── case-studies/       # Case study chapters
│   ├── exercises/          # Exercise content
│   └── assets/             # Images and diagrams
├── src/                    # Custom components
│   ├── components/         # React components
│   └── css/               # Custom styles
├── docusaurus.config.js    # Site configuration
├── package.json           # Dependencies and scripts
├── babel.config.js        # Babel configuration
└── static/                # Static assets
```

## Contributing Content

### Adding a New Chapter

1. Create a new directory in the `docs/` folder:
   ```bash
   mkdir docs/new-chapter-name
   ```

2. Add your content as Markdown files:
   ```bash
   touch docs/new-chapter-name/index.md
   ```

3. Edit the file with your content following the standard format:

```markdown
---
title: Chapter Title
sidebar_label: Chapter Name
sidebar_position: 1
description: Brief description of the chapter
keywords: [list, of, keywords]
---

# Chapter Title

## Learning Objectives

- Objective 1
- Objective 2
- Objective 3

## Section Title

Your content here...

## Exercises

1. **Exercise Title**: Brief description of the exercise.

## Summary

Brief summary of the chapter content.
```

### Adding Visual Aids

1. Place images in the `docs/assets/` directory
2. Reference them in your Markdown using relative paths:
   ```markdown
   ![Caption text](./assets/image-name.png)
   ```

### Adding Code Examples

Use Markdown code blocks with appropriate language specification:

```markdown
## Example Code

Here's an example of a robot control algorithm:

\`\`\`python
def move_robot_to_position(robot, target_position):
    """Move robot to target position using inverse kinematics."""
    joint_angles = calculate_inverse_kinematics(target_position)
    robot.set_joint_positions(joint_angles)
    return robot.get_current_position()
\`\`\`
```

## Content Guidelines

### Writing Style

- Write at Flesch-Kincaid grade level 12-14
- Use clear, concise language
- Define technical terms when first used
- Include practical examples with theoretical concepts
- Ensure content is accessible to all learners

### Citation Format

Use APA format for all citations:

```markdown
According to Smith et al. (2023), the approach is effective for humanoid robotics applications (p. 125).

## References

Smith, J., Johnson, A., & Williams, B. (2023). *Advanced robotics methodologies*. Academic Press.
```

### Quality Standards

- All content must be original or properly attributed
- Zero tolerance for plagiarism
- All claims must be verified through primary sources
- Technical accuracy must be verified by experts
- Include diagrams and visual aids where helpful

## Building and Deployment

### Build the Site

```bash
npm run build
```

### Serve Build Locally

```bash
npm run serve
```

### Deploy to GitHub Pages

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

## Common Tasks

### Adding Navigation

Edit `docusaurus.config.js` to add new pages to the sidebar:

```javascript
sidebar: [
  {
    type: 'category',
    label: 'Chapter Title',
    items: ['chapter-title/page-name'],
  },
],
```

### Adding Custom Components

Create React components in `src/components/` and import them in your Markdown files:

```markdown
import ComponentName from '@site/src/components/ComponentName';

<ComponentName prop="value" />
```

## Getting Help

- Check the [full specification](./spec.md) for detailed requirements
- Review existing chapters for formatting examples
- Contact the project maintainers for complex issues
- Use the [research document](./research.md) for technical guidance