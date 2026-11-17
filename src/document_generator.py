"""
Document Generator for University Academic Regulations Handbook.
Generates an official multi-page PDF with realistic academic regulations,
exact page boundaries, numbered clauses, tables, and headers/footers.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages and prints running header and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1A365D"))
        # Header
        self.drawString(54, 750, "APEX UNIVERSITY - ACADEMIC REGULATIONS & POLIISAS (R-2024)")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 742, letter[0] - 54, 742)
        
        # Footer
        self.line(54, 45, letter[0] - 54, 45)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 32, "Official University Policy Document | Office of the Dean of Academic Affairs")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_text)
        self.restoreState()


def create_handbook_pdf(output_pdf_path: str):
    """Creates a 10-page official academic regulations handbook PDF."""
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=60,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'ChapterHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=8
    )
    
    clause_title_style = ParagraphStyle(
        'ClauseTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=6,
        spaceAfter=3
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("APEX UNIVERSITY ACADEMIC REGULATIONS (R-2024)", title_style))
    story.append(Paragraph("CHAPTER 1: PREAMBLE, PROGRAM NOMENCLATURE & DEGREE DURATIONS", h1_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("Clause 1.1: B.Tech Degree Normal Duration and Credit Requirement", clause_title_style))
    story.append(Paragraph("The Bachelor of Technology (B.Tech) program shall be a four-year full-time undergraduate program comprising eight (8) regular semesters. Every candidate enrolled in the B.Tech program must successfully complete a minimum of one hundred and sixty (160) academic credits as prescribed by the respective Board of Studies. A student shall be deemed eligible for the award of the B.Tech degree only upon clearing all mandatory theory courses, laboratory sessions, internships, and capstone projects.", body_style))
    
    story.append(Paragraph("Clause 1.2: Maximum Duration for Program Completion", clause_title_style))
    story.append(Paragraph("The maximum allowable duration for completing all degree requirements for the 4-year B.Tech program shall be normal duration plus two (2) years, which corresponds to six (6) academic years or twelve (12) consecutive semesters from the date of initial registration. If a student fails to fulfill all academic and credit requirements within this twelve-semester ceiling, their registration shall stand automatically cancelled without further notification.", body_style))
    
    story.append(Paragraph("Clause 1.3: Lateral Entry Students Scheme", clause_title_style))
    story.append(Paragraph("Students admitted directly into the third semester under the Lateral Entry scheme (diploma holders or B.Sc graduates) shall undergo a six-semester program (Semesters 3 through 8). Lateral entry candidates must earn a minimum of one hundred and twenty (120) academic credits. The maximum permissible time for lateral entry candidates to complete degree requirements is five (5) academic years (ten semesters).", body_style))

    story.append(Paragraph("Clause 1.4: Medium of Instruction and Examination", clause_title_style))
    story.append(Paragraph("The official medium of instruction, course delivery, assessments, laboratory assignments, and End Semester Assessments (ESA) shall be strictly English across all academic departments. All project reports, dissertations, and oral defenses must be delivered exclusively in the English language.", body_style))
    
    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    story.append(Paragraph("CHAPTER 2: COURSE REGISTRATION, CREDIT STRUCTURE & ADD/DROP POLICY", h1_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("Clause 2.1: Credit Definition and Calculation System", clause_title_style))
    story.append(Paragraph("Academic credits are computed on a weekly basis per semester: One (1) hour of lecture (L) per week equals 1 credit; one (1) hour of tutorial (T) per week equals 1 credit; two (2) hours of laboratory or practical session (P) per week equals 1 credit. Non-credit mandatory audit courses are designated with zero credits on the grade sheet.", body_style))

    story.append(Paragraph("Clause 2.2: Permissible Semester Credit Load Limits", clause_title_style))
    story.append(Paragraph("A student is required to register for a minimum of 16 credits and a normal maximum of 24 credits in any given regular semester. Under special academic circumstances, students with a cumulative grade point average (CGPA) of 8.50 or above may register for up to 28 credits in a semester, subject to the explicit written approval of the Department Faculty Advisor and the Dean of Academic Affairs.", body_style))

    story.append(Paragraph("Clause 2.3: Course Add/Drop Period and Deadlines", clause_title_style))
    story.append(Paragraph("Students are permitted to modify their course registrations by adding or dropping courses strictly within the first ten (10) instructional days of the commencement of the semester. No request for addition of courses shall be entertained after the tenth instructional day under any circumstances.", body_style))

    story.append(Paragraph("Clause 2.4: Course Withdrawal Policy", clause_title_style))
    story.append(Paragraph("A student may formally withdraw from a registered elective course up to the end of the eighth (8th) instructional week of the semester. A course withdrawn under this provision shall appear on the grade card with the grade letter 'W'. The grade 'W' carries zero grade points and does not impact SGPA or CGPA. Withdrawn courses carry no fee refunds and cannot bring the total active credit load below 16 credits.", body_style))

    story.append(Paragraph("Clause 2.5: Audit Course Registration Guidelines", clause_title_style))
    story.append(Paragraph("Students may register for at most one (1) audit course per semester. Audit courses require at least 75% attendance. Students satisfying attendance shall receive the grade 'AU' on their grade card. Audit courses carry zero grade points and do not count toward cumulative grade calculations.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    story.append(Paragraph("CHAPTER 3: ATTENDANCE POLIISAS, CONDONATION & DETENTION (NSAR)", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 3.1: Mandatory 85% Attendance Threshold", clause_title_style))
    story.append(Paragraph("Every student is required to maintain a minimum attendance of eighty-five percent (85%) of the total instructional hours delivered in each individual registered theory course and practical course in order to become eligible to appear for the End Semester Assessment (ESA) of that course.", body_style))

    story.append(Paragraph("Clause 3.2: Condonation of Attendance Shortage on Valid Grounds", clause_title_style))
    story.append(Paragraph("Condonation of attendance shortage up to a maximum of ten percent (10%)—allowing eligibility for students with attendance between 75.0% and 84.9%—may be granted strictly by the Vice-Chancellor on the recommendation of the Attendance Review Committee. Condonation is permitted solely for valid medical hospitalization or official university deputation (e.g., inter-university sports or academic competitions). The student must submit official medical proof within three (3) working days of returning to campus and pay a non-refundable Condonation Fee of INR 2,000 per course.", body_style))

    story.append(Paragraph("Clause 3.3: Detention under NSAR (Not Satisfied Attendance Requirement)", clause_title_style))
    story.append(Paragraph("Any student whose final attendance in a course falls below 75.0% under any circumstance (including medical illness) shall NOT be eligible for condonation and shall be declared detained in that course under grade code 'NSAR'. The student shall be prohibited from writing the ESA for that course and must mandatorily re-register for the entire course during the summer fast-track semester or subsequent regular semester.", body_style))

    story.append(Paragraph("Clause 3.4: Practical Laboratory Attendance Rules", clause_title_style))
    story.append(Paragraph("One hundred percent (100%) completion of prescribed laboratory experiments and practical write-ups is compulsory. Any student missing a lab session must attend a scheduled compensatory lab session before the end of the semester. Shortage below 85% in laboratory courses is strictly non-condonable.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 4 ====================
    story.append(Paragraph("CHAPTER 4: EVALUATION SCHEME & CONTINUOUS INTERNAL EVALUATION (ISA)", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 4.1: Overall Course Weightage Distribution", clause_title_style))
    story.append(Paragraph("The performance of a student in each academic course is assessed through two main evaluation mechanisms: Internal Semester Assessment (ISA) carrying 50% of total course marks, and End Semester Assessment (ESA) carrying 50% of total course marks.", body_style))

    story.append(Paragraph("Clause 4.2: Components and Breakdown of ISA (50 Marks)", clause_title_style))
    story.append(Paragraph("The 50 marks allocated to Internal Semester Assessment (ISA) are partitioned as follows:", body_style))
    story.append(Paragraph("• Internal Assessment Tests (IAT): Two centralized written tests of 25 marks each conducted across the semester. The aggregate is scaled to 30 marks.", bullet_style))
    story.append(Paragraph("• Quizzes & Surprise Assessments: Two online or in-class objective quizzes of 5 marks each, contributing a total of 10 marks.", bullet_style))
    story.append(Paragraph("• Experiential Learning & Mini-Project Assignment: Open-ended problem-solving assignment or course seminar evaluated for 10 marks.", bullet_style))

    story.append(Paragraph("Clause 4.3: Minimum ISA Threshold for Eligibility to Write ESA", clause_title_style))
    story.append(Paragraph("To be eligible to appear in the End Semester Assessment (ESA) of any course, a student must secure a minimum of forty percent (40%) marks in the Internal Semester Assessment (ISA), which corresponds to at least twenty (20) marks out of fifty (50). Students obtaining less than 20 marks in ISA are categorized as 'NE' (Not Eligible) and shall NOT be permitted to write the ESA.", body_style))

    story.append(Paragraph("Clause 4.4: Laboratory and Practical Course Evaluation", clause_title_style))
    story.append(Paragraph("Laboratory courses are evaluated for 100 marks: 60 marks for continuous weekly observation, journal submission, and viva-voce; 40 marks for the comprehensive final lab practical examination. A minimum score of 50% in aggregate is required to clear laboratory subjects.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 5 ====================
    story.append(Paragraph("CHAPTER 5: END SEMESTER ASSESSMENT (ESA) & GRADING SYSTEM", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 5.1: ESA Blueprint and Duration", clause_title_style))
    story.append(Paragraph("The End Semester Assessment (ESA) for theory courses is a three-hour closed-book centralized written examination set for 100 marks and scaled down to 50 marks. Question papers contain five distinct modules with internal choice within each module.", body_style))

    story.append(Paragraph("Clause 5.2: Minimum Passing Standards in a Course", clause_title_style))
    story.append(Paragraph("A candidate shall be declared to have passed a course if and only if they obtain: (i) at least 40% marks in the End Semester Assessment (minimum 20 out of 50 scaled marks, or 40 out of 100 raw marks), AND (ii) an aggregate of at least 40% marks combined across ISA and ESA (minimum 40 marks out of 100).", body_style))

    story.append(Paragraph("Clause 5.3: Grading Scale and Grade Points Table", clause_title_style))
    
    # Grading Table
    grade_data = [
        ["Percentage Marks Range", "Letter Grade", "Grade Description", "Grade Point"],
        ["90% to 100%", "O", "Outstanding", "10"],
        ["80% to 89%", "S", "Excellent", "9"],
        ["70% to 79%", "A", "Very Good", "8"],
        ["60% to 69%", "B", "Good", "7"],
        ["50% to 59%", "C", "Fair", "6"],
        ["40% to 49%", "D", "Pass", "5"],
        ["Below 40%", "F", "Fail", "0"],
        ["Not Eligible / Detained", "NE / NSAR", "Detained", "0"],
        ["Incomplete / Absent", "I / AB", "Incomplete", "0"]
    ]
    t = Table(grade_data, colWidths=[140, 90, 150, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white])
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Clause 5.4: SGPA and CGPA Mathematical Formulations", clause_title_style))
    story.append(Paragraph("The Semester Grade Point Average (SGPA) is computed as: SGPA = Sum(C_i * G_i) / Sum(C_i), where C_i represents course credits and G_i represents grade points earned. The Cumulative Grade Point Average (CGPA) is computed across all registered courses from the first semester up to the current semester in identical fashion.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 6 ====================
    story.append(Paragraph("CHAPTER 6: ACADEMIC PROGRESSION, DETAINMENT & MINIMUM CREDITS", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 6.1: Promotion Criteria to 3rd Semester (Second Year)", clause_title_style))
    story.append(Paragraph("To be eligible for promotion from the first year to the third semester (commencement of 2nd year), a student must have earned at least 60% of the total cumulative credits prescribed for the first and second semesters combined (i.e., at least 24 credits out of 40 first-year credits).", body_style))

    story.append(Paragraph("Clause 6.2: Promotion Criteria to 5th Semester (Third Year) - Zero Backlog Rule", clause_title_style))
    story.append(Paragraph("To be promoted to the fifth semester (commencement of 3rd year), a student must fulfill two concurrent criteria: (i) Must have successfully cleared ALL courses of the first year (1st and 2nd semesters) with zero pending backlogs, AND (ii) Must have earned at least 60% of the total credits prescribed for the second year (3rd and 4th semesters).", body_style))

    story.append(Paragraph("Clause 6.3: Promotion Criteria to 7th Semester (Fourth Year)", clause_title_style))
    story.append(Paragraph("To register for the seventh semester (final year), a candidate must: (i) have passed ALL courses of the second year (3rd and 4th semesters) with zero pending backlogs, AND (ii) have earned at least 60% of the total credits prescribed for the third year (5th and 6th semesters).", body_style))

    story.append(Paragraph("Clause 6.4: Year Detention Policy", clause_title_style))
    story.append(Paragraph("Any student failing to satisfy the credit promotion or zero-backlog prerequisite rules specified in Clauses 6.1 through 6.3 shall be declared Year-Detained. A detained student cannot attend higher semester classes and must spend an academic year clearing backlog courses before progressing.", body_style))

    story.append(Paragraph("Clause 6.5: Minimum CGPA Requirement for Degree Award", clause_title_style))
    story.append(Paragraph("A student must secure a cumulative CGPA of not less than 5.00 at the conclusion of the eighth semester to be eligible for the award of B.Tech Degree. Candidates securing a CGPA between 4.00 and 4.99 must register for grade improvement in the summer semester to attain the 5.00 threshold.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 7 ====================
    story.append(Paragraph("CHAPTER 7: EXAMINATION MALPRACTICE, CODE OF CONDUCT & PENALTIES", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 7.1: Classification of Examination Malpractice Offences", clause_title_style))
    story.append(Paragraph("Academic dishonesty during any quiz, ISA, or ESA is subject to strict penalties. Prohibited acts include: carrying written notes/chits, electronic gadgets, programmable calculators, smart watches, mobile phones into exam halls, copying, exchange of answer booklets, and impersonation.", body_style))

    story.append(Paragraph("Clause 7.2: Category I Penalty - Unauthorized Possession Without Copying", clause_title_style))
    story.append(Paragraph("Possession of unauthorized study materials, manuscripts, formula sheets, or mobile devices in turned-off state without evidence of copying: Penalty entails immediate confiscation of material, cancellation of the examination in that specific course, and award of 'F' grade for that course.", body_style))

    story.append(Paragraph("Clause 7.3: Category II Penalty - Active Copying or Digital Transmission", clause_title_style))
    story.append(Paragraph("Active copying from unauthorized material, receiving or transmitting answers via smartphone/smartwatch, or assisting other examinees: Penalty entails cancellation of all examinations registered in the current semester, award of 'F' grade in all current courses, and debarment for one (1) subsequent regular semester.", body_style))

    story.append(Paragraph("Clause 7.4: Category III Penalty - Impersonation and Severe Misconduct", clause_title_style))
    story.append(Paragraph("Impersonation of an examinee, assault or threat to invigilators, or intentional destruction of answer scripts: Immediate expulsion of the candidate from the university for two (2) academic years, cancellation of past semester results, and lodging of a formal police First Information Report (FIR).", body_style))

    story.append(Paragraph("Clause 7.5: Examination Disciplinary Committee (EDC) Jurisdiction", clause_title_style))
    story.append(Paragraph("All reported malpractice cases are adjudicated by the five-member Examination Disciplinary Committee (EDC) chaired by the Dean of Student Affairs. The student is afforded an opportunity for a personal hearing within seven (7) working days of the incident before the final penalty is ratified.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 8 ====================
    story.append(Paragraph("CHAPTER 8: RE-EVALUATION, SCRIPT INSPECTION & CHALLENGE VALUATION", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 8.1: Photostat Copy of Answer Booklet Inspection", clause_title_style))
    story.append(Paragraph("A student desiring to review their evaluated ESA answer script may apply for a certified photocopy within five (5) working days from the date of online publication of results. The application fee is INR 500 per course (non-refundable). Scripts are made available digitally within 3 working days.", body_style))

    story.append(Paragraph("Clause 8.2: Re-evaluation Procedure and Deadlines", clause_title_style))
    story.append(Paragraph("Students may apply for re-evaluation of theory answer scripts within ten (10) working days of result declaration upon payment of a re-evaluation fee of INR 1,200 per course. Re-evaluation is conducted through a double-blind independent assessment by a subject expert outside the teaching faculty.", body_style))

    story.append(Paragraph("Clause 8.3: Mark Revision and Fee Refund Rules", clause_title_style))
    story.append(Paragraph("If the marks obtained in re-evaluation exceed the original marks by five percent (5%) or more of the maximum examination marks, the re-evaluated marks shall replace the original marks. If marks increase by fifteen percent (15%) or more, the script is sent to a third independent evaluator whose score is final, and the entire re-evaluation fee of INR 1,200 shall be refunded to the student. If the change is less than 5%, the original marks remain unaltered.", body_style))

    story.append(Paragraph("Clause 8.4: Challenge Valuation Scheme", clause_title_style))
    story.append(Paragraph("If a student remains aggrieved following re-evaluation, they may apply for Challenge Valuation within seven (7) days of re-evaluation results by depositing INR 5,000. Evaluation is conducted jointly by a panel of two senior professors in the presence of the Controller of Examinations.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 9 ====================
    story.append(Paragraph("CHAPTER 9: MAKE-UP EXAMS, SUMMER FAST-TRACK & REMEDIAL SCHEME", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 9.1: Eligibility Criteria for Make-up Examinations", clause_title_style))
    story.append(Paragraph("Make-up examinations are NOT supplementary or backlog exams. Make-up exams are permissible strictly for students who were absent from the regular ESA due to documented medical emergency hospitalization or official national representation, and who possess at least 60% in ISA. Students with 'F', 'NE', or 'NSAR' grades are completely barred from make-up exams.", body_style))

    story.append(Paragraph("Clause 9.2: Timing and Grade Capping in Make-up Exams", clause_title_style))
    story.append(Paragraph("Make-up examinations are conducted within two (2) weeks following the announcement of regular ESA results. The maximum letter grade awarded in a make-up examination shall be capped at grade 'A' (Grade Point 8), irrespective of whether the numerical score falls in the 'S' or 'O' range.", body_style))

    story.append(Paragraph("Clause 9.3: Summer Fast-Track Semester Structure", clause_title_style))
    story.append(Paragraph("A special summer fast-track semester of six (6) to eight (8) weeks duration is conducted during June and July. A student may register for a maximum of two (2) theory courses and one (1) laboratory course (totaling not more than ten [10] credits) for re-registration or backlog clearance.", body_style))

    story.append(Paragraph("Clause 9.4: Remedial Support System", clause_title_style))
    story.append(Paragraph("Departments conduct mandatory remedial tutorial sessions on Saturdays for students scoring below 50% in ISA. Remedial attendance is tracked by the Academic Mentoring Cell and is a prerequisite for receiving borderline condonation.", body_style))

    story.append(PageBreak())

    # ==================== PAGE 10 ====================
    story.append(Paragraph("CHAPTER 10: HONOURS, MINORS, CURRICULAR REVISIONS & INTERNSHIPS", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Clause 10.1: B.Tech with Honours Degree Requirements", clause_title_style))
    story.append(Paragraph("An undergraduate student can earn a B.Tech with Honours by meeting the following strict criteria: (i) Must possess a cumulative CGPA of 8.50 or higher at the end of the fourth semester with zero backlogs; (ii) Must register for and successfully complete an additional eighteen to twenty (18-20) advanced departmental credits across semesters 5 through 8; (iii) Must maintain a CGPA of >= 8.50 until graduation without any grade 'F' throughout the degree.", body_style))

    story.append(Paragraph("Clause 10.2: B.Tech with Minor Degree Requirements", clause_title_style))
    story.append(Paragraph("To obtain a Minor in an interdisciplinary engineering discipline, a student with a minimum CGPA of 7.00 may earn eighteen (18) additional credits offered by the secondary department. Minor courses must be completed alongside regular coursework within the prescribed eight semesters.", body_style))

    story.append(Paragraph("Clause 10.3: Mandatory Industrial Internship Guidelines", clause_title_style))
    story.append(Paragraph("Every student must undergo a mandatory full-time industrial internship of a minimum duration of six (6) weeks during the summer vacation between the sixth (6th) and seventh (7th) semesters. The internship carries three (3) academic credits, evaluated in the seventh semester through an industrial mentor evaluation, written technical report, and an oral viva-voce defense.", body_style))

    story.append(Paragraph("Clause 10.4: Final Year Capstone Project Structure", clause_title_style))
    story.append(Paragraph("The eighth semester is dedicated to the Capstone Major Project carrying twelve (12) credits. Projects are evaluated by an external and internal committee in three phases: Phase I (Problem Formulation & Literature Review - 20%), Phase II (Mid-term Prototype - 30%), and Final Defense & Demonstration (50%).", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated official 10-page handbook PDF at: {output_pdf_path}")


def generate_text_representation(pdf_path: str, txt_path: str):
    """Generates a text mirror with explicit page delimiter tags."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    lines = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        lines.append(f"--- [PAGE {page_num}] ---")
        lines.append(text.strip())
        lines.append(f"--- [END OF PAGE {page_num}] ---\n")
    
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved text mirror at: {txt_path}")


if __name__ == "__main__":
    pdf_out = "data/raw_documents/university_academic_regulations_2024.pdf"
    txt_out = "data/raw_documents/academic_regulations_text.txt"
    create_handbook_pdf(pdf_out)
    generate_text_representation(pdf_out, txt_out)
