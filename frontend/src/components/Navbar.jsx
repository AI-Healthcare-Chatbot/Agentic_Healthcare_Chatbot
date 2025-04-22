import React, { useState } from 'react';

const Navbar = () => {
  // Replace this URL with your actual GitHub repository URL
  const githubRepoUrl = "https://github.com/AI-Healthcare-Chatbot/Agentic_Healthcare_Chatbot";
  
  // State for controlling the info modal
  const [showInfoModal, setShowInfoModal] = useState(false);

  // Function to toggle the info modal
  const toggleInfoModal = () => {
    setShowInfoModal(!showInfoModal);
  };

  return (
    <>
      <header className="bg-white shadow-sm py-4">
        <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center">
            <svg 
              className="h-8 w-8 text-blue-600" 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              strokeWidth={1.8} 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" 
              />
            </svg>
            <h1 className="ml-2 text-xl font-bold text-gray-900">
              Healthcare Chatbot
            </h1>
          </div>

          <div className="hidden sm:flex items-center space-x-8">
            {/* Info Button */}
            <button
              onClick={toggleInfoModal}
              className="flex items-center text-gray-700 hover:text-blue-600"
            >
              <svg 
                className="w-5 h-5 mr-1" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                strokeWidth={2} 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" 
                />
              </svg>
              Info
            </button>

            {/* Resources Dropdown */}
            <div className="relative group">
              <button className="flex items-center text-gray-700 hover:text-blue-600">
                <span>Resources</span>
                <svg 
                  className="ml-1 w-4 h-4" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={2} 
                  stroke="currentColor"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M19.5 8.25l-7.5 7.5-7.5-7.5" 
                  />
                </svg>
              </button>
              <div className="absolute z-10 left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300">
                <a 
                  href="https://www.cdc.gov/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  CDC
                </a>
                <a 
                  href="https://www.who.int/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  WHO
                </a>
                <a 
                  href="https://www.nih.gov/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  NIH
                </a>
              </div>
            </div>

            {/* Documentation Link */}
            <a 
              href="https://github.com/yourusername/your-repo-name"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-700 hover:text-blue-600"
            >
              Documentation
            </a>

            {/* GitHub Link - Updated with your repo URL */}
            <a 
              href={githubRepoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-gray-700 hover:text-blue-600"
            >
              <svg 
                className="w-5 h-5 mr-1" 
                xmlns="http://www.w3.org/2000/svg" 
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              GitHub
            </a>
          </div>

          {/* Mobile menu button */}
          <button className="sm:hidden text-gray-700 hover:text-blue-600">
            <svg 
              className="w-6 h-6" 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              strokeWidth={2} 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" 
              />
            </svg>
          </button>
        </div>
      </header>

      {/* Info Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto p-6 shadow-xl">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-gray-900">About Our Data Source</h2>
              <button
                onClick={toggleInfoModal}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg
                  className="w-6 h-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            
            <div className="text-gray-700 space-y-4">
              <h3 className="font-semibold text-lg">General Description</h3>
              <p>
                New York State Planning and Research Cooperative System (SPARCS) is a comprehensive data reporting system established in 1979 as a result of cooperation between the health care industry and government. Initially created to collect information on discharges from hospitals, SPARCS currently collects patient level detail on patient characteristics, diagnoses and treatments, services, and charges for every hospital discharge, ambulatory surgery, outpatient services and emergency department visit in New York State.
              </p>
              
              <p>
                The enabling legislation and regulations for SPARCS are located under Section 28.16 of the Public Health Law (PHL), Section 400.18 of Title 10 (Health) of the Official Compilation of Codes, Rules, and Regulations of the State of New York (NYCRR). More information on SPARCS may be found on the <a href="http://www.health.ny.gov/statistics/sparcs/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">New York State Department of Health's website</a>.
              </p>
              
              <h3 className="font-semibold text-lg">Data Privacy</h3>
              <p>
                A goal of the Department is to provide the public with easy-to-use tools to answer health care questions while protecting the privacy of the individual patient. Protecting the privacy of the patient in the collected data is a primary function of the Department of Health SPARCS program staff. SPARCS has assigned personnel whose responsibility is the monitoring of the release and handling of the data.
              </p>
              
              <p>
                SPARCS utilizes information technology protections for the collection, storage, and release of data. SPARCS data files are available with and without identifying data elements. A multi-tiered approach in the release of data is utilized to provide patient privacy while still providing useful health care data for the public and researchers.
              </p>
              
              <h3 className="font-semibold text-lg">Data Methodology</h3>
              <p>
                Any facility certified to provide Article 28 inpatient services, ambulatory surgery services, emergency department services or outpatient services is required to submit data to SPARCS. Submitting facilities include New York State Hospitals and Diagnostic and Treatment Centers (D&TCs, commonly known as clinics). This includes both hospital-owned and operated, as well as free-standing D&TC facilities. Regardless of their ownership, each facility must report data for each specific facility. More information on how SPARCS data is collected may be found at: <a href="http://www.health.ny.gov/statistics/sparcs/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">SPARCS Data Collection</a>.
              </p>
              
              <h3 className="font-semibold text-lg">Contact Information</h3>
              <p>
                New York State Department of Health<br />
                Office of Health Services Quality and Analytics<br />
                Empire State Plaza<br />
                Corning Tower, Room 1938<br />
                Albany, New York 12237<br />
                Phone: (518) 473-8144<br />
                Fax: (518) 486-3518<br />
                E-mail: <a href="mailto:SPARCS.submissions@health.ny.gov" className="text-blue-600 hover:underline">SPARCS.submissions@health.ny.gov</a> or <a href="mailto:SPARCS.requests@health.ny.gov" className="text-blue-600 hover:underline">SPARCS.requests@health.ny.gov</a>
              </p>
            </div>
            
            <div className="mt-6">
              <button
                onClick={toggleInfoModal}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Navbar;