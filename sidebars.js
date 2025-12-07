// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro/index',
        'intro/humanoid-robotics-basics',
        'intro/physical-ai-basics',
        'intro/case-study',
        'intro/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Introduction to Physical AI & Humanoid Robotics',
        description: 'Getting started with the fundamentals',
        slug: '/intro'
      }
    },
    {
      type: 'category',
      label: 'Theoretical Foundations',
      items: [
        'theoretical-foundations/index',
        'theoretical-foundations/mathematical-foundations',
        'theoretical-foundations/control-theory',
        'theoretical-foundations/case-study',
        'theoretical-foundations/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Theoretical Foundations',
        description: 'Mathematical and theoretical concepts',
        slug: '/theoretical-foundations'
      }
    },
    {
      type: 'category',
      label: 'Practical Implementation',
      items: [
        'practical-implementation/index',
        'practical-implementation/ros-setup',
        'practical-implementation/simulation',
        'practical-implementation/code-samples',
        'practical-implementation/case-study',
        'practical-implementation/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Practical Implementation',
        description: 'Hands-on implementation guides',
        slug: '/practical-implementation'
      }
    },
    {
      type: 'category',
      label: 'Control Systems',
      items: [
        'control-systems/index',
        'advanced-control/index',
        'advanced-control/case-study',
        'advanced-control/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Control Systems',
        description: 'Understanding robot control mechanisms',
        slug: '/control-systems'
      }
    },
    {
      type: 'category',
      label: 'Learning Algorithms',
      items: [
        'learning-algorithms/index',
        'ml-robotics/index',
        'ml-robotics/case-study',
        'ml-robotics/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Learning Algorithms',
        description: 'AI and machine learning for humanoid robots',
        slug: '/learning-algorithms'
      }
    },
    {
      type: 'category',
      label: 'Human-Robot Interaction',
      items: [
        'hri/index',
        'hri/case-study',
        'hri/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Human-Robot Interaction',
        description: 'Designing effective human-robot interactions',
        slug: '/hri'
      }
    },
    {
      type: 'category',
      label: 'Sensor Integration',
      items: [
        'sensor-integration/index',
        'sensor-integration/case-study',
        'sensor-integration/exercises'
      ],
      link: {
        type: 'generated-index',
        title: 'Sensor Integration',
        description: 'Integrating multiple sensors for humanoid robots',
        slug: '/sensor-integration'
      }
    },
    {
      type: 'category',
      label: 'Ethics and Safety',
      items: [
        'ethics-safety/index'
      ],
      link: {
        type: 'generated-index',
        title: 'Ethics and Safety',
        description: 'Responsible development and deployment',
        slug: '/ethics-safety'
      }
    },
    {
      type: 'category',
      label: 'Case Studies',
      items: [
        'case-studies/index'
      ],
      link: {
        type: 'generated-index',
        title: 'Case Studies',
        description: 'Real-world applications and examples',
        slug: '/case-studies'
      }
    },
    {
      type: 'category',
      label: 'Future Directions and Summary',
      items: [
        'future-directions/index'
      ],
      link: {
        type: 'generated-index',
        title: 'Future Directions and Summary',
        description: 'Looking ahead and key takeaways',
        slug: '/future-directions'
      }
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'glossary',
        'content-guidelines',
        'educator-resources'
      ],
      link: {
        type: 'generated-index',
        title: 'Resources',
        description: 'Additional materials and resources',
        slug: '/resources'
      }
    }
  ],
};

module.exports = sidebars;