 %%
 %% lincolncsuthesis.cls
 %% Copyright (c) 2020 Benjamin Williams <bwilliams@lincoln.ac.uk>
 %%
 %% A LaTeX template for typesetting undergraduate degree theses
 %% according to the University of Lincoln School of Computer Science
 %% thesis guidelines.
 %%
 %% It is worth noting that the main inspiration (and small snippets of code) 
 %% for this template was Cameron Gray's bangorcsthesis template.
 %%
 %% God bless you Cameron, you absolute crazy genius.
 %%  
 %% This template is also heavily influenced by the original 
 %% lincolncsthesis class, for postgraduate research degrees at
 %% the UoL SoCS (also written by me).
 %%
 %%
 %% -----------------------------------------
 %%
 %% This work may be distributed and/or modified under the
 %% conditions of the LaTeX Project Public License, either version 1.3
 %% of this license of (at your option) any later version.
 %% The latest version of this license is in
 %% http://www.latex-project.org/lppl.txt
 %% and version 1.3 or later is part of all distributions of LaTeX
 %% version 2005/12/01 or later.
 %%


% We're gonna be defining variables as long globals
% (because they might wanna split these up with extra lines)
\def\lxdef{\long\xdef}

% This class name
\def\@classname {lincolncsuthesis}


% Changelog (implemented 12/6/2020)
% -------------
% - 1.0.1 [12/6/2020]: Added appendices and altered PGR template for undergraduate use.
% - 1.0.2 [26/10/2020]: Fixed appendices environment and numbering. Altered info text slightly. Added titletoc option to appendices package.


% METADATA AND OPTIONS
% --------------------

% Define thesis template version and other metadata
\gdef\@templateVersion{1.0.2}
\gdef\@templateDate{2020/26/10}
\gdef\@templateAuthor{Benjamin Williams}

% It provides this class
\ProvidesClass{\@classname}[\@templateDate \@templateVersion \@templateAuthor]

% We need LaTeX2e at least
\NeedsTeXFormat{LaTeX2e}

% Article class
\LoadClass[12pt]{report}




% PACKAGES
% ------------

% Packages
\RequirePackage{amsmath}
\RequirePackage{amssymb}
% --
\RequirePackage{ifthen}
\RequirePackage{xkeyval}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage[T1]{fontenc} % font types and character verification
\RequirePackage[UKenglish]{babel}
\RequirePackage[UKenglish]{isodate}
\RequirePackage[utf8]{inputenc}
\RequirePackage{xparse}
\RequirePackage{graphicx}
\RequirePackage{longtable}
\RequirePackage{blindtext}
\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage{geometry}
\RequirePackage{layout}
\RequirePackage{courier}
\RequirePackage{subcaption}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage[table,figure]{totalcount}
\RequirePackage[titletoc]{appendix}
\RequirePackage{framed}
% -----------------
\RequirePackage{csquotes}
\RequirePackage{titlesec}
\RequirePackage{datetime2}
\RequirePackage{fancyhdr}
\RequirePackage{lmodern}
% -----------------



% Define variables relevant to this thesis
\lxdef \@logo           {FHD_logo}
\newcommand{\thesisLogoPath}[1]{\xdef\@logo{#1}}
\lxdef \@degree         {M.Eng}
\newcommand{\thesisDegree}[1]{\xdef\@degree{#1}}
\lxdef \@programme      {Embedded systems for Mechatronics}
\newcommand{\thesisProgramme}[1]{\xdef\@programme{#1}}
\lxdef \@schoolname     {School of Inforamtics}
\newcommand{\thesisSchool}[1]{\xdef\@schoolname{#1}}
\lxdef \@universityname {Dortmund University of Applied Sciences and Arts}
\newcommand{\thesisUniversity}[1]{\xdef\@universityname{#1}}
\lxdef \@studentnumber {7206937}
\newcommand{\thesisStudentNumber}[1]{\xdef\@studentnumber{#1}}
\lxdef \@headercontents {Assessment Item}
\newcommand{\thesisHeaderContents}[1]{\xdef\@headercontents{#1}}
% -----
\newcommand{\thesisTurnOffHF}{\xdef\@headerturnoff{yup}}
% -----




% AFTER TITLE PAGE BUT BEFORE THESIS CONTENT
% ------------------------------------------
\newcommand{\afterTitlePage}{

    % Set margins for the rest of the thesis. UoL guidelines 
    % state that: [left >= 40mm], [others >= 15mm]
    \newgeometry{
        %nohead, 
        headheight=5cm,
        headsep=0.5cm,
        footskip=1cm,
        top=25mm,
        bottom=2.5cm,
        right=25mm,
        left=40mm,
        marginparwidth=0cm,
        marginparsep=0cm
    }
    
    % Set spacing and reset numbering
    \pagenumbering{roman}
            
    % Set double spacing (from cleanthesis.sty)
    \setstretch{1.5}
    \clubpenalty = 10000
    \widowpenalty = 10000
    \displaywidowpenalty = 10000
}


\newcommand{\beforeTitlePage}{
    % Set margins for the rest of the thesis. UoL guidelines 
    % state that: [left >= 40mm], [others > 15mm]
    \newgeometry{
        nohead, 
        top=40mm,
        bottom=25mm,
        right=30mm,
        left=40mm,
        marginparwidth=0cm,
        marginparsep=0cm
    }
}



% Print out submission text
\newcommand{\submissionStatement}{
    \setstretch{1.25}
    \@ifundefined{@submissiontext}{Submitted in partial satisfaction of the requirements for the \\degree of \\ 
        \vspace{1cm}\@degree~\@programme}{\@submissiontext}
}

% The default title page: title, then logo, then author
\newcommand{\@titlepage}{
    \begin{titlepage}
    \topskip0pt
    \vspace*{\fill}

    % Make the title
    \makeatletter
    \centering{\LARGE \@title \par} \vspace{1.5cm}
    
    % Make the logo
    \includegraphics[height=1.5in]{\@logo}\par 
    
    % Some space, then the author, then the school and university
    % name
    \vspace{1.5cm}
    {\large \@author \@ifundefined{@studentnumber}{}{\\ \vspace{0.25cm} \@studentnumber} \par}\vspace{0.45cm}

    % Some space and the submission statement
    {\vspace{1cm}}
    {\footnotesize \submissionStatement \par}

    {\footnotesize \@schoolname \\  \vspace{0.25cm}  \@universityname \par}

    % Print out the supervisor
    \vspace{1cm}
    
    % Then the fancy date!
    {\footnotesize \@ifundefined{@modulecode}{}{\@modulecode \\} \@date \par}
    \vspace*{\fill}
    %
	\end{titlepage}
}



% The default title page style is the title, logo, author style
\def \@defaultTitlePage {\@titlepage}


% Lets renew the make title command so we don't have to 
% make a custom one
\RenewDocumentCommand{\maketitle}{o}{%

  % Call function for before title page
  \beforeTitlePage
  
  % Then check which title page style to use
 % \IfNoValueTF{#1}
    {\@defaultTitlePage}
    %{\ifthenelse{\equal{#1}{logo-first}}
      %  {\@altTitlePage}
       % {\@defaultTitlePage}}
    
  % Call function for after title page       
  \afterTitlePage
}




% ACKNOWLEDGEMENTS
% ----------------

\newenvironment{acknowledgements}
    {\cleardoublepage
     \chapter*{Acknowledgements}
    }
    {}
    
    
    

% ABSTRACT
% -------------------------

\renewenvironment{abstract}
    {\cleardoublepage
     \chapter*{\abstractname}
    }
    {}
    
    
% TABLE OF CONTENTS, FIGURES AND TABLES
% ---------------------------

\newcommand{\thesisTOF}{
    \iftotalfigures
        \cleardoublepage
        \listoffigures
    \fi
}

\newcommand{\thesisTOT}{
    \iftotaltables
        \cleardoublepage
        \listoftables
    \fi
}

\newcommand{\thesisTables}{
    \cleardoublepage
    \renewcommand{\contentsname}{Table of Contents}
    \setcounter{tocdepth}{3}
    {\setstretch{1.25}
        \tableofcontents
        \thesisTOF
        \thesisTOT
    }
    
    \setlength{\abovedisplayskip}{1pt}
    \setlength{\belowdisplayskip}{15pt}
}




% START OF THESIS
% --------------

\fancypagestyle{plain}{
    \renewcommand{\headrulewidth}{0pt}%
    \fancyhf{}%
    \fancyfoot[R]{\footnotesize\textbf{\thepage}}%    
    
    \@ifundefined{@headerturnoff}{
        \fancyhead[R]{\footnotesize \textit{\@headercontents}}
	    \fancyfoot[L]{\footnotesize \textit{\@author~(\@studentnumber)}}
    }{}
}


\fancypagestyle{mainStyle}{
	\renewcommand{\headrulewidth}{0pt}%
	\fancyhf{}%
	\fancyfoot[R]{\footnotesize\emph{\leftmark} \hspace{0.25cm} \footnotesize\textbf{\thepage}}%   


	\fancyhead[R]{\footnotesize\textit{\@headercontents}}
	\fancyfoot[L]{\footnotesize \textit{\@author~(\@studentnumber)}}
}

\newcommand{\thesisBodyStart}{

    % Have we turned off headers? If so, kill the foot + head
    \@ifundefined{@headerturnoff}{}{
        \fancyfoot[L]{~}
        \fancyhead[R]{~}
    }
    
    % Clear the page so we don't number the last page
    \cleardoublepage
    
    % Set page numbering back to 1, 2, 3 .. 
    \pagenumbering{arabic} 
}






% REFERENCING
% -----------

% Fake harvard referencing, instead of using natbib -- because
% natbib makes biblatex blow up.
\def\@importHarvard {
    \RequirePackage[
        defernumbers=true,
        backend=biber,
        bibencoding=utf8,
        natbib=true,
        hyperref=true,
        backref=true,
        urldate=long,
        style=authoryear,
        citestyle=authoryear,
        sorting=nyt
    ]{biblatex}
}

% Import it
\@importHarvard

% Command for printing out references list
\newcommand{\printReferences}[1][References]{
	
	% Set some spacing up between the items
	\setlength\bibitemsep{0.35cm}
	
	% Set the line height (of actual items) to 1.0
	\setstretch{1.0}
	
	% Print out the bibliography with the given title
	\printbibliography[notkeyword={game}, title=#1]
}


% TITLE / CHAPTERS / FIGURES / TITLESEC SPACING
% ---------------------------------------------

% Set the chapter title style
\titleformat{\chapter}[display]
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{0pt}{\Huge}

% And space it out a bit
\titlespacing*{\chapter} {0pt}{0pt}{20mm}

% A fake chapter title command to emulate chapter titles
% without actually adding a chapter
\newcommand{\chapterTitle}[1]{
	\vspace*{-20mm} 
	{\normalfont\huge\bfseries#1 \par} 
	\vspace*{20mm}
}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{\setstretch{0.5}}



% LUDOGRAPHY
% ----------


% Citation environment for games
\defbibenvironment{games}
 {\list
     {}
     {\setlength{\leftmargin}{\bibhang}%
      \setlength{\itemindent}{-\leftmargin}%
      \setlength{\itemsep}{\bibitemsep}%
      \setlength{\parsep}{\bibparsep}}}
  {\endlist}
  {\item}
  
% Prints out the ludography (all citations with the
% key "game")
\newcommand{\printLudography}[1][Ludography]{

    % Set some spacing up between the items
	\setlength\bibitemsep{0.35cm}
	
	% Set the line height (of actual items) to 1.0
	\setstretch{1.0}
	
	% Print out the bibliography with the given title
	\printbibliography[env=games, keyword={game}, title=#1]
}

% Prints out the ludography without a heading
\newcommand{\printLudographyNoHead}{

    % Set some spacing up between the items
	\setlength\bibitemsep{0.35cm}
	
	% Set the line height (of actual items) to 1.0
	\setstretch{1.0}
	
	% Print out the bibliography with the given title
	\printbibliography[heading=none, env=games, keyword={game}]
}

% Ludography environment, if you wish to put some text before
% the list of games that are printed
\newenvironment{ludography}[1][Ludography]{
  \chapter*{#1}  
}{\vspace*{1cm} \printLudographyNoHead}

% For citing games
\newcommand{\citeGame}[1]{
    \emph{\citeauthor{#1}}
}



% BLANK PAGE / QUOTE PAGE
% -----------------------

% For quote pages, copyright notices, or whatever else
\newenvironment{blankpage}[1][\unskip]{%
	
	% Check if [c] is specified, if so, then 
	% make it vertically align.. if not, then
	% just clear a double page and write some stuff
    \ifthenelse{\equal{#1}{c}}{
        \cleardoublepage
        \topskip0pt
        \vspace*{\fill}
    }{
    \cleardoublepage
    \topskip0pt
    \vspace*{1in}
    }
}{
        \vspace*{\fill}
}





% Set the page style to the main style. Set up the chapter
% mark style so that it shows the title of the chapter only.
\pagestyle{mainStyle}
\renewcommand{\chaptermark}[1]{\markboth{{#1}}{}}