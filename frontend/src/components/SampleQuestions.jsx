import React, { useContext } from 'react';
import { ChatContext } from '../contexts/ChatContext';

const SampleQuestions = () => {
  const { sendMessage } = useContext(ChatContext);

  const questions = [
    {
      category: 'Healthcare Insights',
      samples: [
        "What are the top 5 diagnoses for patients over 70?",
        "Show me the most expensive procedures in hospitals",
        "What's the average length of stay for heart surgery patients?",
        "Compare readmission rates between rural and urban hospitals"
      ]
    },
    {
      category: 'Visualizations',
      samples: [
        "Create a pie chart of patient demographics by age group",
        "Generate a bar chart of the most common diagnoses",
        "Plot the average cost by hospital type",
        "Visualize the distribution of severity levels across different age groups"
      ]
    },
    {
      category: 'Medical Information',
      samples: [
        "What are the symptoms of pneumonia?",
        "What is the difference between type 1 and type 2 diabetes?",
        "What treatments are available for hypertension?",
        "What are common side effects of statins?"
      ]
    }
  ];

  const handleQuestionClick = (question) => {
    sendMessage(question);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 mb-4">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Sample Questions</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {questions.map((group, groupIndex) => (
          <div key={groupIndex} className="space-y-2">
            <h3 className="text-sm font-medium text-gray-700 border-b pb-1 mb-2">
              {group.category}
            </h3>
            <ul className="space-y-2">
              {group.samples.map((question, questionIndex) => (
                <li key={questionIndex}>
                  <button
                    onClick={() => handleQuestionClick(question)}
                    className="text-left text-sm text-blue-600 hover:text-blue-800 hover:underline w-full"
                  >
                    {question}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SampleQuestions;