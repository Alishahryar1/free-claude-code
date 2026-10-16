"use strict";

const STORAGE_KEY = "gestion-scolaire-data-v1";
const FIRST_PASSWORD = "79408093Ms@";
const MAX_BACKUP_BYTES = 64_000_000;
const SAMPLE_LOGO = `data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><rect width="96" height="96" rx="22" fill="#183a4d"/><path d="M20 39 48 23l28 16v34H20z" fill="#dce8e4"/><path d="M31 45h34v25H31z" fill="#fff"/><path d="M48 45v25M31 53h34M31 61h34" fill="none" stroke="#183a4d" stroke-width="3"/><path d="m17 40 31-19 31 19" fill="none" stroke="#c57b53" stroke-linecap="round" stroke-linejoin="round" stroke-width="6"/></svg>')}`;
const MALI_FLAG = '<svg class="mali-flag" viewBox="0 0 36 24" role="img" aria-label="Drapeau du Mali"><rect width="12" height="24" fill="#14b53a"/><rect x="12" width="12" height="24" fill="#fcd116"/><rect x="24" width="12" height="24" fill="#ce1126"/></svg>';
const DEFAULT_CARD_TEMPLATE = { layout: "horizontal", showPhoto: true, showMatricule: true, showBirthDate: true, showFamilyPhone: true, showGuardian: true };
const ACTIONS = ["read", "create", "update", "delete"];
const ACADEMIC_CYCLES = {
  "Crèche": ["Petit", "Grand"],
  "Primaire": ["1re", "2e", "3e", "4e", "5e", "6e"],
  "Secondaire": ["7e", "8e", "9e"],
  "Lycée": ["10e CM", "11e LL", "11e SES", "11e S", "12e TLL", "12e TAL", "12e TSS", "12e TSE", "12e TSEExp", "12e TSEco"],
};
const NEXT_LEVELS = {
  Petit: ["Grand"], Grand: ["1re"], "1re": ["2e"], "2e": ["3e"], "3e": ["4e"], "4e": ["5e"], "5e": ["6e"], "6e": ["7e"], "7e": ["8e"], "8e": ["9e"], "9e": ["10e CM"],
  "10e CM": ["11e LL", "11e SES", "11e S"], "11e LL": ["12e TLL", "12e TAL"], "11e SES": ["12e TSEco"], "11e S": ["12e TSS", "12e TSE", "12e TSEExp"],
};
const TERMINAL_LEVELS = new Set(["12e TLL", "12e TAL", "12e TSS", "12e TSE", "12e TSEExp", "12e TSEco"]);
const LEGACY_CLASS_LEVELS = { CP1: "1re", CE1: "2e", CE2: "3e", CM1: "4e", CM2: "5e" };
const ACADEMIC_LEVEL_ALIASES = { "12e TSExp": "12e TSEExp", "12e TEXP": "12e TSEExp", TEXP: "12e TSEExp" };
const makeSubjects = (items) => items.map(([name, optional = false]) => ({ name, coefficient: 1, optional }));
const DEFAULT_CURRICULUM = Object.fromEntries([
  ...["1re", "2e", "3e", "4e", "5e", "6e"].map((level) => [level, makeSubjects([
    ["Français · lecture"], ["Français · expression orale"], ["Français · expression écrite"],
    ["Français · grammaire"], ["Français · orthographe"], ["Français · conjugaison"],
    ["Français · vocabulaire"], ["Mathématiques"], ["EPS"], ["ECM"],
    ["Éducation artistique"], ["APP · initiation au milieu"], ["Langue nationale", true],
  ])]),
  ...["7e", "8e", "9e"].map((level) => [level, makeSubjects([
    ["Français"], ["Mathématiques"], ["Physique-chimie"], ["SVT"],
    ["Histoire-géographie"], ["Anglais"], ["ECM"], ["EPS"],
    ["Dessin · éducation artistique"], ["Arabe", true], ["Langue nationale", true],
  ])]),
  ["10e CM", makeSubjects([
    ["Français"], ["Mathématiques"], ["Physique-chimie"], ["SVT"],
    ["Histoire-géographie"], ["Anglais"], ["LV2 au choix", true],
    ["ECM"], ["EPS"],
  ])],
  ["11e LL", makeSubjects([["Français · littérature"], ["Philosophie"], ["Anglais"], ["LV2"], ["Histoire-géographie"], ["Mathématiques"], ["ECM"], ["EPS"]])],
  ["11e SES", makeSubjects([["Français"], ["Mathématiques"], ["Économie"], ["Sciences sociales"], ["Histoire-géographie"], ["Anglais"], ["ECM"], ["EPS"]])],
  ["11e S", makeSubjects([["Français"], ["Mathématiques"], ["Physique-chimie"], ["SVT"], ["Anglais"], ["Philosophie"], ["ECM"], ["EPS"]])],
  ["12e TLL", makeSubjects([["Français · littérature"], ["Philosophie"], ["Anglais"], ["LV2"], ["Histoire-géographie"], ["Langue nationale"], ["ECM"], ["EPS"]])],
  ["12e TAL", makeSubjects([["Français · littérature"], ["Arabe"], ["Philosophie"], ["Anglais"], ["Histoire-géographie"], ["LV2"], ["ECM"], ["EPS"]])],
  ["12e TSS", makeSubjects([["Français"], ["Philosophie"], ["Histoire-géographie"], ["Mathématiques · statistiques"], ["Sciences sociales"], ["Anglais"], ["ECM"], ["EPS"]])],
  ["12e TSE", makeSubjects([["Français"], ["Mathématiques"], ["Physique-chimie"], ["SVT"], ["Anglais"], ["Philosophie"], ["ECM"], ["EPS"]])],
  ["12e TSEExp", makeSubjects([["Français"], ["Mathématiques"], ["Physique-chimie"], ["SVT"], ["Anglais"], ["Philosophie"], ["ECM"], ["EPS"]])],
  ["12e TSEco", makeSubjects([["Français"], ["Mathématiques"], ["Économie"], ["Comptabilité-gestion"], ["Statistiques"], ["Histoire-géographie"], ["Anglais"], ["Philosophie"], ["ECM"], ["EPS"]])],
].map(([level, subjects]) => [level, subjects]));
const MODULES = {
  dashboard: "Tableau de bord",
  enrollments: "Inscriptions",
  students: "Élèves",
  classes: "Classes",
  dossiers: "Dossiers élèves",
  teachers: "Enseignants & contrats",
  payments: "Paiements",
  expenses: "Dépenses",
  accounting: "Comptabilité",
  grades: "Notes & bulletins",
  curriculum: "Matières & coefficients",
  attendance: "Présences",
  discipline: "Conduite & incidents",
  bulletins: "Bulletins",
  documents: "Documents imprimables",
  cards: "Cartes scolaires",
  accounts: "Comptes & accès",
  settings: "Établissement",
  audit: "Journal d’audit",
};
const METHODS = ["Espèces", "Orange Money", "Moov Money", "Wave", "Virement bancaire"];
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const uid = () => crypto.randomUUID();
const frDate = (value) => new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "short", year: "numeric" }).format(new Date(`${value}T12:00:00`));
const today = () => new Date().toISOString().slice(0, 10);
const daysAgo = (days) => {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().slice(0, 10);
};
const schoolYear = () => {
  const date = new Date();
  const start = date.getMonth() >= 7 ? date.getFullYear() : date.getFullYear() - 1;
  return `${start} — ${start + 1}`;
};
const money = (value) => new Intl.NumberFormat("fr-FR", { style: "currency", currency: "XOF", maximumFractionDigits: 0 }).format(Number(value) || 0);
const esc = (value = "") => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const initials = (name = "") => name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "É";
const defaultPermissions = () => Object.fromEntries(Object.keys(MODULES).map((key) => [key, { read: true, create: true, update: true, delete: true }]));
function daysUntil(value) {
  if (!value) return null;
  return Math.ceil((new Date(`${value}T12:00:00`) - new Date(`${today()}T12:00:00`)) / 86400000);
}
function enrollmentForYear(student, year) {
  const history = (student.enrollmentHistory || []).find((item) => item.schoolYear === year);
  if (history) return history;
  if (student.enrollmentYear === year || (!student.enrollmentYear && year === data.school.year)) {
    return { schoolYear: year, className: student.className };
  }
  return null;
}
function studentsInClass(className, year = data.school.year) {
  return data.students.filter((student) => {
    const enrollment = enrollmentForYear(student, year);
    return enrollment?.className === className;
  }).sort((a, b) => `${a.lastName} ${a.firstName}`.localeCompare(`${b.lastName} ${b.firstName}`, "fr"));
}
function academicLevel(schoolClass) {
  const value = schoolClass.academicLevel || LEGACY_CLASS_LEVELS[schoolClass.name] || "";
  return ACADEMIC_LEVEL_ALIASES[value] || value;
}
function normalizeAcademicCatalog(school) {
  const cycles = school.academicCycles || ACADEMIC_CYCLES;
  school.academicCycles = Object.fromEntries(Object.entries(cycles).map(([cycle, levels]) => [
    cycle,
    [...new Set(levels.map((level) => ACADEMIC_LEVEL_ALIASES[level] || level))],
  ]));
  const curriculum = school.curriculum || {};
  school.curriculum = { ...structuredClone(DEFAULT_CURRICULUM) };
  for (const [level, subjects] of Object.entries(curriculum)) {
    const canonical = ACADEMIC_LEVEL_ALIASES[level] || level;
    if (!curriculum[canonical] || level === canonical) school.curriculum[canonical] = subjects;
  }
}
function subjectsForLevel(level) {
  const normalized = ACADEMIC_LEVEL_ALIASES[level] || level;
  return data.school.curriculum?.[normalized] || DEFAULT_CURRICULUM[normalized] || [];
}
function coefficientForClassSubject(className, subjectName) {
  const schoolClass = data.classes.find((item) => item.name === className);
  return Number(subjectsForLevel(academicLevel(schoolClass || {})).find((item) => item.name === subjectName)?.coefficient) || 1;
}
function curriculumEditorMarkup() {
  const canEdit = can("curriculum", "update") || can("settings", "update");
  const levels = Object.entries(data.school.academicCycles || ACADEMIC_CYCLES).flatMap(([, items]) => items);
  if (!levels.includes(selectedCurriculumLevel)) selectedCurriculumLevel = levels[0] || "1re";
  const subjects = subjectsForLevel(selectedCurriculumLevel);
  return `<section class="settings-section"><h2>Matières & coefficients</h2><p>Référentiel de départ par niveau/série. Coefficient 1 est une valeur neutre provisoire : adaptez chaque matière aux coefficients officiels de l’établissement; les calculs pondérés utilisent les valeurs enregistrées.</p><label class="field"><span class="field-label">Niveau ou série</span><select class="field-control" id="curriculumLevel">${Object.entries(data.school.academicCycles || ACADEMIC_CYCLES).flatMap(([, levels]) => levels).map((level) => `<option value="${esc(level)}" ${level === selectedCurriculumLevel ? "selected" : ""}>${esc(academicLabel(level))}</option>`).join("")}</select></label><div class="table-wrap"><table><thead><tr><th>Matière</th><th>Coefficient</th><th>Optionnelle</th><th></th></tr></thead><tbody>${subjects.map((subject, index) => `<tr><td><input class="field-control curriculum-subject" data-curriculum-subject="${index}" value="${esc(subject.name)}" aria-label="Nom de la matière" ${canEdit ? "" : "disabled"}></td><td><input class="field-control curriculum-coefficient" type="number" min="0.25" step="0.25" data-curriculum-coefficient="${index}" value="${esc(subject.coefficient ?? 1)}" aria-label="Coefficient de ${esc(subject.name)}" ${canEdit ? "" : "disabled"}></td><td><input type="checkbox" data-curriculum-optional="${index}" ${subject.optional ? "checked" : ""} aria-label="Matière optionnelle" ${canEdit ? "" : "disabled"}></td><td>${canEdit ? `<button class="icon-button" data-action="remove-curriculum-subject" data-index="${index}" aria-label="Retirer ${esc(subject.name)}">×</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="4">Aucune matière configurée.</td></tr>`}</tbody></table></div>${canEdit ? btn("Ajouter une matière", "add-curriculum-subject", "button button-secondary", "+") : ""}<p class="small-print muted">Les options de langue sont marquées comme facultatives. Saisissez ici les coefficients propres à chaque niveau et série.</p></section>`;
}
function readCurriculumEditor() {
  const rows = $$("#pageContent tbody tr").filter((row) => row.querySelector("[data-curriculum-subject]"));
  const subjects = rows.map((row) => {
    const name = row.querySelector("[data-curriculum-subject]").value.trim();
    const coefficient = Number(row.querySelector("[data-curriculum-coefficient]").value);
    if (!name || !Number.isFinite(coefficient) || coefficient <= 0) throw new Error("Chaque matière doit avoir un nom et un coefficient positif.");
    return { name, coefficient, optional: row.querySelector("[data-curriculum-optional]").checked };
  });
  if (new Set(subjects.map((subject) => subject.name.toLocaleLowerCase("fr"))).size !== subjects.length) {
    throw new Error("Une matière ne peut apparaître qu’une fois dans un même niveau.");
  }
  return subjects;
}
function syncCurriculumDraft() {
  const level = $("#curriculumLevel")?.value || selectedCurriculumLevel;
  const rows = $$("#pageContent tbody tr").filter((row) => row.querySelector("[data-curriculum-subject]"));
  data.school.curriculum[level] = rows.map((row) => ({
    name: row.querySelector("[data-curriculum-subject]").value.trim(),
    coefficient: Number(row.querySelector("[data-curriculum-coefficient]").value),
    optional: row.querySelector("[data-curriculum-optional]").checked,
  }));
}
function academicLabel(level) {
  const normalized = ACADEMIC_LEVEL_ALIASES[level] || level;
  return normalized === "12e TSEExp" ? "12e TSEExp · TEXP" : normalized;
}
function conductCapital() {
  return [18, 20].includes(Number(data.school.conductMaximum)) ? Number(data.school.conductMaximum) : 18;
}
function incidentsFor(studentId) {
  return data.incidents.filter((incident) => incident.studentId === studentId);
}
function conductBalance(studentId) {
  const computed = data.analytics?.studentFinancials?.[studentId]?.conductBalance;
  if (apiMode && Number.isFinite(computed)) return computed;
  const deducted = incidentsFor(studentId)
    .filter((incident) => incident.validated)
    .reduce((sum, incident) => sum + Math.max(0, Number(incident.pointsRemoved) || 0), 0);
  return Math.max(0, conductCapital() - deducted);
}
function annualAverageFor(studentId, year) {
  const computed = data.analytics?.averages?.[`${studentId}|${year}|annual`]?.average;
  if (apiMode) return Number.isFinite(computed) ? computed : null;
  const marks = data.grades.filter((grade) => {
    if (grade.studentId !== studentId || !Number.isFinite(grade.score)) return false;
    const assessment = data.assessments.find((item) => item.id === grade.assessmentId);
    return (grade.schoolYear || assessment?.schoolYear || data.school.year) === year;
  });
  const weight = marks.reduce((sum, grade) => sum + Number(grade.coefficient || 1), 0);
  return weight ? marks.reduce((sum, grade) => sum + Number(grade.score) * Number(grade.coefficient || 1), 0) / weight : null;
}
function yearsInData() {
  return [...new Set([
    data.school.year,
    ...(data.school.academicClosures || []).map((item) => item.schoolYear),
    ...data.students.flatMap((student) => (student.enrollmentHistory || []).map((item) => item.schoolYear)),
    ...data.grades.map((grade) => grade.schoolYear),
    ...data.assessments.map((assessment) => assessment.schoolYear),
  ].filter(Boolean))].sort((a, b) => b.localeCompare(a));
}
function contractStatus(teacher) {
  if (teacher.status !== "Actif") return [teacher.status || "Inactif", "inactive"];
  if (!teacher.endDate) return ["Sans échéance", ""];
  const remaining = daysUntil(teacher.endDate);
  if (remaining < 0) return ["Contrat échu", "overdue"];
  if (remaining <= 30) return ["À renouveler", "pending"];
  return ["En cours", ""];
}
function followUps() {
  const items = [];
  data.students.filter((student) => student.active && outstanding(student) > 0).slice(0, 4).forEach((student) => {
    items.push({ title: `${student.firstName} ${student.lastName}`, detail: `${student.className} · solde à payer ${money(outstanding(student))}`, type: "student", id: student.id, severity: "pending" });
  });
  data.students.filter((student) => student.active && (!student.guardian || !student.phone)).slice(0, 2).forEach((student) => {
    items.push({ title: `${student.firstName} ${student.lastName}`, detail: "Fiche incomplète · responsable ou téléphone manquant", type: "student", id: student.id, severity: "overdue" });
  });
  data.teachers.filter((teacher) => {
    const remaining = daysUntil(teacher.endDate);
    return teacher.status === "Actif" && remaining !== null && remaining <= 30;
  }).forEach((teacher) => {
    items.push({ title: `${teacher.firstName} ${teacher.lastName}`, detail: contractStatus(teacher)[0] === "Contrat échu" ? "Contrat arrivé à échéance" : `Contrat à renouveler le ${fmtDate(teacher.endDate)}`, type: "teacher", id: teacher.id, severity: "overdue" });
  });
  return items;
}
function permissionsForRole(role) {
  const permissions = Object.fromEntries(Object.keys(MODULES).map((key) => [key, { read: false, create: false, update: false, delete: false }]));
  permissions.dashboard.read = true;
  if (role === "proviseur") return defaultPermissions();
  if (role === "admin") {
    const all = defaultPermissions();
    ["payments", "expenses", "accounting", "audit"].forEach((module) => { all[module].delete = false; });
    all.accounts.activate = true;
    return all;
  }
  if (role === "censeur") {
    permissions.students.read = true;
    permissions.classes.read = true;
    permissions.dossiers.read = true;
    permissions.teachers = { read: true, create: true, update: true, delete: true };
    permissions.grades = { read: true, create: true, update: true, delete: true };
    permissions.discipline = { read: true, create: true, update: true, delete: true };
    permissions.bulletins = { read: true, create: true, update: true, delete: true };
    permissions.curriculum = { read: true, create: true, update: true, delete: true };
    permissions.documents.read = true;
    permissions.cards.read = true;
  } else if (role === "comptable") {
    permissions.students.read = true;
    permissions.payments = { read: true, create: true, update: true, delete: true };
    permissions.expenses = { read: true, create: true, update: true, delete: true };
    permissions.accounting = { read: true, create: true, update: true, delete: true };
    permissions.audit.read = true;
  } else if (role === "econome") {
    permissions.enrollments = { read: true, create: true, update: true, delete: false };
    permissions.students = { read: true, create: true, update: true, delete: false };
    permissions.payments = { read: true, create: true, update: true, delete: false };
    permissions.expenses = { read: true, create: true, update: true, delete: false };
    permissions.accounting = { read: true, create: true, update: true, delete: false };
    permissions.audit.read = true;
  } else if (role === "surveillant") {
    permissions.students = { read: true, create: false, update: true, delete: false };
    permissions.classes = { read: true, create: true, update: true, delete: true };
    permissions.attendance = { read: true, create: true, update: true, delete: true };
    permissions.dossiers.read = true;
    permissions.documents.read = true;
  } else if (["directeur_primaire", "directeur_secondaire"].includes(role)) {
    ["dashboard", "students", "classes", "dossiers", "teachers", "grades", "bulletins", "curriculum", "attendance", "documents", "discipline"].forEach((module) => {
      permissions[module] = { read: true, create: true, update: true, delete: true };
    });
    permissions.students.delete = false;
    permissions.enrollments = { read: true, create: true, update: true, delete: false };
  } else if (role === "enseignant") {
    permissions.students.read = true;
    permissions.classes.read = true;
    permissions.dossiers.read = true;
    permissions.documents.read = true;
    permissions.grades = { read: true, create: true, update: true, delete: false };
    permissions.bulletins.read = true;
    permissions.bulletins.create = true;
  } else if (role === "accueil") {
    permissions.enrollments = { read: true, create: true, update: false, delete: false };
    permissions.students = { read: true, create: true, update: false, delete: false };
    permissions.classes.read = true;
    permissions.dossiers.read = true;
  } else {
    permissions.students.read = true;
    permissions.classes.read = true;
    permissions.dossiers.read = true;
  }
  return permissions;
}
function dossierMarkup() {
  const student = data.students.find((item) => item.id === selectedDossierId) || data.students[0];
  if (!student) return `${pageHeading("Dossiers élèves", "Dossier individuel et suivi automatisé.")}<section class="surface empty-state"><strong>Aucun dossier à afficher</strong>Créez une inscription pour commencer le suivi.</section>`;
  selectedDossierId = student.id;
  const studentPayments = [...data.payments].filter((item) => item.studentId === student.id).sort((a, b) => b.date.localeCompare(a.date));
  const marks = marksFor(student.id);
  const average = averageFor(student.id);
  const tasks = [];
  if (outstanding(student) > 0) tasks.push(`Solde de scolarité à recouvrer : ${money(outstanding(student))}.`);
  if (!student.phone || !student.guardian) tasks.push("Compléter le nom du responsable et son numéro de téléphone.");
  if (!student.birthDate) tasks.push("Date de naissance non renseignée.");
  if (!marks.length) tasks.push(`Aucune note saisie pour le ${selectedGradeTerm.toLowerCase()}.`);
  return `${pageHeading("Dossiers élèves", "Informations, paiements, résultats et rappels calculés depuis la fiche.", `${btn("Imprimer le dossier complet", "print-dossier", "button button-primary", "⎙")}${canRead("grades") ? `<button class="button button-secondary" data-action="print-evaluation-history" data-id="${student.id}">Suivi des évaluations</button>` : ""}`)}
    <div class="inline-filters dossier-picker"><label class="field-label" for="dossierStudent">Choisir un élève</label><select id="dossierStudent" class="select-control">${data.students.map((item) => `<option value="${item.id}" ${item.id === student.id ? "selected" : ""}>${esc(item.firstName)} ${esc(item.lastName)} · ${esc(item.className)}</option>`).join("")}</select><span class="status ${studentStatus(student)[1]}">${studentStatus(student)[0]}</span></div>
    <section class="surface student-profile"><div class="profile-photo">${student.photo ? `<img src="${esc(student.photo)}" alt="Photo de ${esc(student.firstName)} ${esc(student.lastName)}">` : `<span>${esc(initials(`${student.firstName} ${student.lastName}`))}</span>`}</div><div class="profile-main"><h2>${esc(student.firstName)} ${esc(student.lastName)}</h2><p>${esc(student.className)} · Inscrit(e) le ${fmtDate(student.enrollmentDate)}</p><div class="profile-details"><div><span>Date de naissance</span><strong>${fmtDate(student.birthDate)}</strong></div><div><span>Responsable légal</span><strong>${esc(student.guardian || "À compléter")}</strong></div><div><span>Téléphone</span><strong>${esc(student.phone || "À compléter")}</strong></div><div><span>Frais annuels</span><strong>${money(student.annualFee)}</strong></div>${canRead("discipline") ? `<div><span>Note de conduite</span><strong>${conductBalance(student.id)} / ${conductCapital()}</strong></div>` : ""}</div></div><div class="profile-balance"><span>RESTE À PAYER</span><strong>${money(outstanding(student))}</strong><small>${money(totalPaid(student.id))} réglés sur ${money(student.annualFee)}</small></div></section>
    <div class="dashboard-lower dossier-sections"><section class="surface"><div class="surface-header"><div><h2>Suivi & actions à traiter</h2><p>Rappels calculés automatiquement à partir de la fiche.</p></div></div>${tasks.length ? `<ul class="follow-up-list">${tasks.map((task) => `<li><span class="follow-up-marker"></span>${esc(task)}</li>`).join("")}</ul>` : `<div class="empty-state">Aucun suivi à traiter. La fiche est à jour.</div>`}</section>
    <section class="surface"><div class="surface-header"><div><h2>Résultats du trimestre</h2><p>${esc(selectedGradeTerm)} · moyenne pondérée</p></div><span class="grade-average">${average === null ? "—" : `${average.toFixed(2)} / 20`}</span></div><div class="table-wrap"><table><thead><tr><th>Matière / évaluation</th><th>Note</th><th>Coefficient</th></tr></thead><tbody>${marks.length ? marks.map((grade) => `<tr><td>${esc(grade.subject)}<small class="muted"><br>${esc(grade.type || "Évaluation")}</small></td><td>${Number(grade.score).toFixed(2)} / 20</td><td>${Number(grade.coefficient) || 1}</td></tr>`).join("") : `<tr><td colspan="3">Aucune note enregistrée.</td></tr>`}</tbody></table></div></section></div>
    ${canRead("discipline") ? `<section class="surface dossier-payments"><div class="surface-header"><div><h2>Suivi de conduite</h2><p>Capital : ${conductCapital()} points · seuls les retraits validés sont appliqués; aucune pénalité n’est prédéfinie.</p></div>${can("discipline", "create") ? `<button class="button button-secondary" data-action="new-incident" data-id="${student.id}">Signaler un incident</button>` : ""}</div><div class="table-wrap"><table><thead><tr><th>Date</th><th>Motif & observation</th><th>Retrait demandé</th><th>Statut</th><th>Auteur / validation</th></tr></thead><tbody>${incidentsFor(student.id).sort((a, b) => b.date.localeCompare(a.date)).map((incident) => `<tr><td>${fmtDate(incident.date)}</td><td><strong>${esc(incident.reason)}</strong><small class="muted"><br>${esc(incident.observation || "—")}</small></td><td>${Number(incident.pointsRemoved)} pts</td><td><span class="status ${incident.validated ? "" : "pending"}">${incident.validated ? "Validé" : "En attente"}</span></td><td>${esc(incident.author || "—")}${incident.validated ? `<small class="muted"><br>Validé par ${esc(incident.validatedBy || "—")}</small>` : ""}</td></tr>`).join("") || `<tr><td colspan="5">Aucun incident enregistré.</td></tr>`}</tbody></table></div></section>` : ""}
    <section class="surface dossier-payments"><div class="surface-header"><div><h2>Historique des paiements</h2><p>Tous les versements rattachés à ce dossier.</p></div>${can("payments", "create") ? `<button class="button button-secondary" data-action="student-payment" data-id="${student.id}">Enregistrer un paiement +</button>` : ""}</div><div class="table-wrap"><table><thead><tr><th>Reçu</th><th>Date</th><th>Moyen de paiement</th><th>Montant</th></tr></thead><tbody>${studentPayments.length ? studentPayments.map((item) => `<tr><td>${esc(item.reference || "—")}</td><td>${fmtDate(item.date)}</td><td>${esc(item.method)}</td><td>${money(item.amount)}</td></tr>`).join("") : `<tr><td colspan="4">Aucun versement enregistré.</td></tr>`}</tbody></table></div></section>`;
}
function teachersMarkup() {
  const teachers = [...data.teachers].sort((a, b) => a.lastName.localeCompare(b.lastName, "fr"));
  const rows = teachers.length ? teachers.map((teacher) => {
    const [status, cls] = contractStatus(teacher);
    const classNames = data.classes.filter((item) => item.teacherId === teacher.id).map((item) => item.name);
    return `<tr><td>${personCell({ firstName: teacher.firstName, lastName: teacher.lastName, gender: "M", guardian: teacher.subject })}</td><td>${esc(teacher.subject)}</td><td>${esc(classNames.join(", ") || "Aucune classe affectée")}</td><td>${esc(teacher.phone || "Contact manquant")}</td><td>${esc(teacher.contractType)}</td><td>${fmtDate(teacher.startDate)}</td><td>${teacher.endDate ? fmtDate(teacher.endDate) : "—"}</td><td>${money(teacher.salary)}</td><td><span class="status ${cls}">${esc(status)}</span></td><td class="screen-only"><div class="table-actions">${can("teachers", "update") ? `<button class="icon-button" data-action="edit-teacher" data-id="${teacher.id}" aria-label="Modifier ${esc(teacher.firstName)} ${esc(teacher.lastName)}">✎</button>` : ""}${can("teachers", "delete") ? `<button class="icon-button" data-action="delete-teacher" data-id="${teacher.id}" aria-label="Supprimer l’enseignant">×</button>` : ""}</div></td></tr>`;
  }).join("") : `<tr><td colspan="8"><div class="empty-state"><strong>Aucun enseignant</strong>Ajoutez une fiche et un contrat pour commencer le suivi.</div></td></tr>`;
  return `${pageHeading("Enseignants & contrats", "Le Censeur suit les affectations, les contrats et leurs échéances.", `${buttonForPermission("Ajouter un enseignant", "new-teacher", "teachers", "button button-primary", "+")}${btn("Imprimer la liste", "print", "button button-secondary", "⎙")}`)}<section class="surface"><div class="surface-header"><div><h2>Personnel enseignant</h2><p>${teachers.filter((teacher) => teacher.status === "Actif").length} enseignants actifs · ${teachers.filter((teacher) => contractStatus(teacher)[0] === "À renouveler" || contractStatus(teacher)[0] === "Contrat échu").length} contrats à suivre</p></div></div><div class="table-wrap"><table><thead><tr><th>Enseignant</th><th>Matière</th><th>Classes</th><th>Contact WhatsApp</th><th>Contrat</th><th>Début</th><th>Échéance</th><th>Salaire mensuel</th><th>Statut</th><th class="screen-only">Actions</th></tr></thead><tbody>${rows}</tbody></table></div></section><p class="small-print muted">Les rappels de fin de contrat sont calculés automatiquement à partir de la date d’échéance. Affectez chaque professeur à une classe depuis la fiche de classe; son contact WhatsApp sert à préparer les listes nominatives.</p>`;
}
function disciplineMarkup() {
  const incidents = [...data.incidents].sort((a, b) => b.date.localeCompare(a.date));
  const pending = incidents.filter((incident) => !incident.validated).length;
  return `${pageHeading("Conduite & incidents", "Registre d’incidents, validation des retraits et suivi des points.", `${can("discipline", "create") ? btn("Signaler un incident", "new-incident", "button button-primary", "+") : ""}${btn("Imprimer le registre", "print", "button button-secondary", "⎙")}`)}
    <section class="surface"><div class="surface-header"><div><h2>Note de conduite</h2><p>Capital actuel : ${conductCapital()} points. Le retrait est saisi au cas par cas; aucune pénalité automatique n’est appliquée.</p></div><span class="status ${pending ? "pending" : ""}">${pending} en attente de validation</span></div>
      <div class="table-wrap"><table><thead><tr><th>Élève / classe</th><th>Date</th><th>Motif</th><th>Points demandés</th><th>Solde validé</th><th>État</th><th>Auteur</th><th class="screen-only">Actions</th></tr></thead><tbody>${incidents.length ? incidents.map((incident) => { const student = studentById(incident.studentId); return `<tr><td>${student ? `${esc(student.firstName)} ${esc(student.lastName)} · ${esc(student.className)}` : "Dossier élève indisponible"}</td><td>${fmtDate(incident.date)}</td><td>${esc(incident.reason)}<small class="muted"><br>${esc(incident.observation || "—")}</small></td><td>${Number(incident.pointsRemoved)} / ${conductCapital()}</td><td>${student ? `${conductBalance(student.id)} / ${conductCapital()}` : "—"}</td><td><span class="status ${incident.validated ? "" : "pending"}">${incident.validated ? "Validé" : "En attente"}</span></td><td>${esc(incident.author || "—")}${incident.validated ? `<small class="muted"><br>${esc(incident.validatedBy || "—")} · ${fmtDate((incident.validatedAt || "").slice(0, 10))}</small>` : ""}</td><td class="screen-only"><div class="table-actions">${!incident.validated && can("discipline", "update") ? `<button class="button button-quiet" data-action="validate-incident" data-id="${incident.id}">Valider le retrait</button>` : ""}${!incident.validated && can("discipline", "delete") ? `<button class="icon-button" data-action="delete-incident" data-id="${incident.id}" aria-label="Supprimer le signalement en attente">×</button>` : ""}</div></td></tr>`; }).join("") : `<tr><td colspan="8"><div class="empty-state"><strong>Aucun incident enregistré</strong>Les retraits ne s’appliquent qu’après validation par un compte habilité.</div></td></tr>`}</tbody></table></div></section>
    <p class="small-print muted">Les retraits validés sont inscrits dans l’audit et verrouillés pour préserver l’historique. Le solde de chaque élève est plafonné à zéro. Un auteur habilité doit saisir lui-même le nombre de points à retirer.</p>`;
}
function attendanceMarkup() {
  const classes = data.classes.filter((item) => item.active);
  if (!classes.some((item) => item.name === selectedAttendanceClass)) {
    selectedAttendanceClass = classes[0]?.name || "";
  }
  const roster = selectedAttendanceClass ? studentsInClass(selectedAttendanceClass).filter((student) => student.active) : [];
  const rows = roster.map((student) => {
    const entry = data.attendance.find((item) => item.studentId === student.id && item.date === selectedAttendanceDate);
    return `<tr><td>${esc(student.lastName)} ${esc(student.firstName)}<small class="muted"><br>${esc(student.matricule || "")}</small></td><td><select class="field-control attendance-status screen-only" data-attendance-status="${esc(student.id)}" aria-label="Présence de ${esc(student.firstName)} ${esc(student.lastName)}" ${can("attendance", "update") || can("attendance", "create") ? "" : "disabled"}><option value="" ${!entry ? "selected" : ""}>À renseigner</option>${["Présent", "Absent", "Retard", "Excusé"].map((status) => `<option value="${status}" ${entry?.status === status ? "selected" : ""}>${status}</option>`).join("")}</select><span class="print-only">${esc(entry?.status || "À renseigner")}</span></td><td><input class="field-control attendance-observation screen-only" data-attendance-observation="${esc(student.id)}" value="${esc(entry?.observation || "")}" aria-label="Observation pour ${esc(student.firstName)} ${esc(student.lastName)}" ${can("attendance", "update") || can("attendance", "create") ? "" : "disabled"}><span class="print-only">${esc(entry?.observation || "—")}</span></td><td>${esc(entry?.recordedBy || "—")}</td></tr>`;
  }).join("");
  const actions = `${can("attendance", "create") || can("attendance", "update") ? btn("Enregistrer les présences", "save-attendance", "button button-primary", "✓") : ""}${btn("Imprimer la feuille", "print", "button button-secondary", "⎙")}`;
  return `${pageHeading("Présences", "Feuille d’appel par classe et par date; chaque saisie reste dans l’historique local.", actions)}
    <section class="surface"><p class="print-only attendance-print-heading">${esc(selectedAttendanceClass)} · ${fmtDate(selectedAttendanceDate)} · ${esc(data.school.name)}</p><div class="inline-filters attendance-filters"><label class="field"><span class="field-label">Classe</span><select class="field-control" id="attendanceClass">${classes.map((item) => `<option value="${esc(item.name)}" ${item.name === selectedAttendanceClass ? "selected" : ""}>${esc(item.name)}</option>`).join("")}</select></label><label class="field"><span class="field-label">Date</span><input class="field-control" id="attendanceDate" type="date" value="${esc(selectedAttendanceDate)}"></label><span class="small-print muted">${roster.length} élève${roster.length === 1 ? "" : "s"}</span></div>
      <div class="table-wrap"><table><thead><tr><th>Élève</th><th>Statut</th><th>Observation</th><th>Saisi par</th></tr></thead><tbody>${rows || `<tr><td colspan="4"><div class="empty-state"><strong>Aucun élève pour cette classe</strong>Choisissez une classe active avec des inscriptions pour établir la feuille d’appel.</div></td></tr>`}</tbody></table></div>
    </section><p class="small-print muted">Les statuts possibles sont présent, absent, retard et excusé. Les données sont conservées dans la base SQLite locale du PC ou, en mode démonstration, dans ce navigateur. Aucune synchronisation cloud n’est activée.</p>`;
}
function curriculumMarkup() {
  return `${pageHeading("Matières & coefficients", "Référentiel pédagogique configurable par niveau et série; le Censeur peut le modifier sans accéder aux paramètres généraux.", `${can("curriculum", "update") ? btn("Enregistrer le référentiel", "save-curriculum", "button button-primary", "✓") : ""}`)}${curriculumEditorMarkup()}`;
}
const roleName = (role) => ({ proviseur: "Proviseur", admin: "Administrateur", censeur: "Censeur", directeur_primaire: "Direction · Crèche & Primaire", directeur_secondaire: "Direction · Secondaire & Lycée", surveillant: "Surveillant", econome: "Économe", comptable: "Comptable", enseignant: "Enseignant", accueil: "Agent d’accueil", user: "Personnel" })[role] || "Personnel";

let data;
let currentUserId = null;
let sessionUser = null;
let apiMode = false;
let remoteWriteQueue = Promise.resolve();
let stateVersion = 0;
let activePage = "dashboard";
let dialogState = null;
let selectedGradeStudentId = null;
let selectedGradeTerm = "1er trimestre";
let selectedDossierId = null;
let selectedAssessmentId = null;
let dialogPhoto = "";
let selectedAcademicYear = null;
let selectedCurriculumLevel = "1re";
let selectedEvaluationYear = null;
let selectedAccountingPeriod = { start: "", end: "" };
let selectedAttendanceClass = "";
let selectedAttendanceDate = today();
let promotionDraft = {};
let toastTimer;

function exampleData() {
  const students = [
    ["Aïssata", "Traoré", "F", "2015-03-14", "CM2", "Mariam Traoré", "70011223"],
    ["Ibrahim", "Diallo", "M", "2016-06-02", "CM1", "Amadou Diallo", "70123456"],
    ["Fatoumata", "Konaté", "F", "2017-01-21", "CE2", "Aminata Konaté", "76004567"],
    ["Mamadou", "Camara", "M", "2014-11-08", "CM2", "Boubacar Camara", "77234567"],
    ["Aminata", "Sow", "F", "2018-05-30", "CE1", "Moussa Sow", "78009876"],
    ["Oumar", "Keïta", "M", "2017-08-17", "CE2", "Awa Keïta", "75008765"],
    ["Mariam", "Coulibaly", "F", "2016-02-09", "CM1", "Issa Coulibaly", "73002211"],
    ["Youssouf", "Diarra", "M", "2015-12-12", "CM2", "Kadiatou Diarra", "79001122"],
  ].map(([first, last, gender, birth, className, guardian, phone], index) => ({
    id: `eleve-${index + 1}`, firstName: first, lastName: last, gender, birthDate: birth,
    className, guardian, phone,     enrollmentDate: daysAgo(18 - index),
    enrollmentYear: schoolYear(), enrollmentHistory: [{ schoolYear: schoolYear(), className, decision: "inscrit" }],
    annualFee: 120000 + (index % 2) * 10000, active: true,
  }));
  const classes = [
    { id: "class-cp1", name: "CP1", level: "Primaire", academicLevel: "1re", capacity: 30, annualFee: 120000, teacher: "Aminata Coulibaly", teacherId: "teacher-1", active: true },
    { id: "class-ce1", name: "CE1", level: "Primaire", academicLevel: "2e", capacity: 30, annualFee: 120000, teacher: "Moussa Traoré", teacherId: "teacher-2", active: true },
    { id: "class-ce2", name: "CE2", level: "Primaire", academicLevel: "3e", capacity: 30, annualFee: 120000, teacher: "Fatoumata Diallo", teacherId: "teacher-3", active: true },
    { id: "class-cm1", name: "CM1", level: "Primaire", academicLevel: "4e", capacity: 30, annualFee: 130000, teacher: "Moussa Traoré", teacherId: "teacher-2", active: true },
    { id: "class-cm2", name: "CM2", level: "Primaire", academicLevel: "5e", capacity: 30, annualFee: 130000, teacher: "Aminata Coulibaly", teacherId: "teacher-1", active: true },
  ];
  const payments = [
    ["eleve-1", 40000, "Orange Money", 12],
    ["eleve-1", 30000, "Wave", 1],
    ["eleve-2", 60000, "Espèces", 11],
    ["eleve-3", 50000, "Virement bancaire", 9],
    ["eleve-4", 90000, "Moov Money", 8],
    ["eleve-5", 30000, "Orange Money", 7],
    ["eleve-6", 45000, "Espèces", 5],
    ["eleve-7", 100000, "Wave", 3],
    ["eleve-8", 65000, "Orange Money", 0],
  ].map(([studentId, amount, method, days], index) => ({ id: `paiement-${index + 1}`, studentId, amount, method, date: daysAgo(days), reference: `REC-${String(104 + index)}`, note: "" }));
  const expenses = [
    { id: "expense-1", label: "Fournitures pédagogiques", category: "Matériel scolaire", amount: 85000, date: daysAgo(10), note: "Cahiers et matériel de classe" },
    { id: "expense-2", label: "Réparation des ventilateurs", category: "Entretien", amount: 42000, date: daysAgo(6), note: "Deux salles de classe" },
    { id: "expense-3", label: "Facture d’électricité", category: "Charges", amount: 57500, date: daysAgo(2), note: "" },
  ];
  const grades = [];
  const gradeSamples = {
    "eleve-1": [["Français", 16], ["Mathématiques", 14], ["Sciences", 15], ["Histoire-Géo", 13]],
    "eleve-2": [["Français", 13], ["Mathématiques", 17], ["Sciences", 14], ["Histoire-Géo", 15]],
    "eleve-3": [["Français", 14], ["Mathématiques", 12], ["Sciences", 16], ["Histoire-Géo", 13]],
    "eleve-4": [["Français", 18], ["Mathématiques", 16], ["Sciences", 15], ["Histoire-Géo", 17]],
    "eleve-5": [["Français", 12], ["Mathématiques", 15], ["Sciences", 13], ["Histoire-Géo", 14]],
    "eleve-6": [["Français", 15], ["Mathématiques", 13], ["Sciences", 14], ["Histoire-Géo", 16]],
    "eleve-7": [["Français", 17], ["Mathématiques", 16], ["Sciences", 18], ["Histoire-Géo", 14]],
    "eleve-8": [["Français", 14], ["Mathématiques", 15], ["Sciences", 16], ["Histoire-Géo", 13]],
  };
  for (const [studentId, subjects] of Object.entries(gradeSamples)) {
    subjects.forEach(([subject, score], index) => grades.push({ id: `${studentId}-grade-${index}`, studentId, subject, score, coefficient: 1, term: "1er trimestre" }));
  }
  return {
    school: { appName: "Gestion scolaire", name: "Groupe scolaire Les Flamboyants", shortName: "Les Flamboyants", address: "Quartier du Fleuve, Bamako", phone: "+223 70 00 00 00", email: "", countryCallingCode: "223", whatsappMode: "test", promotionThreshold: 10, conductMaximum: 18, academicCycles: ACADEMIC_CYCLES, curriculum: structuredClone(DEFAULT_CURRICULUM), color: "#183a4d", secondaryColor: "#dce8e4", accentColor: "#c57b53", logo: SAMPLE_LOGO, year: schoolYear(), academicClosures: [] },
    users: [], students, classes, payments, expenses, grades, teachers: [
      { id: "teacher-1", firstName: "Aminata", lastName: "Coulibaly", subject: "Français", phone: "70001122", contractType: "CDI", startDate: "2023-09-01", endDate: "", salary: 180000, status: "Actif" },
      { id: "teacher-2", firstName: "Moussa", lastName: "Traoré", subject: "Mathématiques", phone: "70002233", contractType: "CDD", startDate: "2025-09-01", endDate: daysAgo(-320), salary: 160000, status: "Actif" },
      { id: "teacher-3", firstName: "Fatoumata", lastName: "Diallo", subject: "Sciences", phone: "70003344", contractType: "CDD", startDate: "2025-09-01", endDate: daysAgo(-21), salary: 155000, status: "Actif" },
    ],
    assessments: [],
    incidents: [],
    attendance: [],
  };
}

function toBase64(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)));
}
function fromBase64(value) {
  return Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
}
async function passwordDigest(password, salt = crypto.getRandomValues(new Uint8Array(16))) {
  if (!crypto?.subtle) throw new Error("Le navigateur ne permet pas de protéger les mots de passe. Ouvrez l’application depuis http://localhost.");
  const material = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits({ name: "PBKDF2", salt, iterations: 150000, hash: "SHA-256" }, material, 256);
  return { salt: toBase64(salt), hash: toBase64(bits) };
}
async function verifyPassword(password, user) {
  const result = await passwordDigest(password, fromBase64(user.salt));
  const a = fromBase64(result.hash);
  const b = fromBase64(user.hash);
  return a.length === b.length && a.every((value, index) => value === b[index]);
}
function persist() {
  if (!apiMode) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      showToast("Enregistrement impossible. Vérifiez l’espace disponible de ce navigateur.");
      console.error("Impossible d’enregistrer les données scolaires.", error);
      throw error;
    }
  }
  if (apiMode && sessionUser) {
    const version = ++stateVersion;
    const { users: ignoredUsers, analytics: ignoredAnalytics, ...state } = JSON.parse(JSON.stringify(data));
    remoteWriteQueue = remoteWriteQueue.then(async () => {
      await apiRequest("/api/state", {
        method: "PUT",
        body: JSON.stringify({ state }),
      });
      try {
        await refreshAnalytics(selectedAccountingPeriod.start, selectedAccountingPeriod.end);
      } catch (error) {
        console.error("Impossible de recalculer les indicateurs scolaires.", error);
        showToast(error.message || "Les indicateurs n’ont pas pu être actualisés.");
      }
    }).catch(async (error) => {
      console.error("Échec de synchronisation avec le serveur scolaire.", error);
      showToast(error.message || "Les changements n’ont pas été enregistrés sur le serveur.");
      if (version !== stateVersion) return;
      try {
        const payload = await apiRequest("/api/state", { cache: "no-store" });
        if (payload.state) {
          data = { ...payload.state, analytics: payload.analytics, users: data.users || [] };
          renderPage();
        }
      } catch (refreshError) {
        console.error("Impossible de recharger les données du serveur.", refreshError);
      }
    });
  }
}
async function prepareStudentPhoto(file) {
  const bitmap = await createImageBitmap(file);
  try {
    const scale = Math.min(1, 640 / bitmap.width, 800 / bitmap.height);
    const canvas = document.createElement("canvas");
    let width = Math.max(1, Math.round(bitmap.width * scale));
    let height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Le navigateur ne peut pas préparer cette photo.");
    while (Math.max(width, height) >= 320) {
      canvas.width = width;
      canvas.height = height;
      context.fillStyle = "#fff";
      context.fillRect(0, 0, width, height);
      context.drawImage(bitmap, 0, 0, width, height);
      let quality = 0.82;
      let photo = canvas.toDataURL("image/jpeg", quality);
      while (photo.length > 400_000 && quality > 0.5) {
        quality = Math.max(0.5, quality - 0.08);
        photo = canvas.toDataURL("image/jpeg", quality);
      }
      if (photo.length <= 400_000) return photo;
      width = Math.round(width * 0.85);
      height = Math.round(height * 0.85);
    }
    throw new Error("La photo reste trop volumineuse après compression.");
  } finally {
    bitmap.close();
  }
}
async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  if (response.status === 204) return {};
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Erreur serveur (${response.status}).`);
  return payload;
}
async function refreshAnalytics(start = "", end = "") {
  if (!apiMode) return;
  const query = new URLSearchParams();
  if (start) query.set("from_date", start);
  if (end) query.set("to_date", end);
  const queryString = query.toString();
  const result = await apiRequest(`/api/analytics${queryString ? `?${queryString}` : ""}`, { cache: "no-store" });
  if (!result.analytics) throw new Error("Les indicateurs scolaires n’ont pas pu être calculés.");
  data.analytics = result.analytics;
}
function currentUser() {
  if (apiMode) return sessionUser;
  return data.users.find((user) => user.id === currentUserId) || null;
}
function can(module, action) {
  const user = currentUser();
  if (!user || !ACTIONS.includes(action)) return false;
  if (user.role === "proviseur") return true;
  return Boolean(user.permissions?.[module]?.[action]);
}
function canRead(module) { return can(module, "read"); }
function moduleForPage(page) {
  return ({ dashboard: "dashboard", eleves: "students", classes: "classes", presences: "attendance", matieres: "curriculum", dossiers: "dossiers", enseignants: "teachers", discipline: "discipline", paiements: "payments", depenses: "expenses", comptabilite: "accounting", bulletins: "grades", cartes: "cards", comptes: "accounts", reglages: "settings", audit: "audit" })[page];
}
function studentById(id) { return data.students.find((student) => student.id === id); }
function studentName(id) {
  const student = studentById(id);
  return student ? `${student.firstName} ${student.lastName}` : "Élève supprimé";
}
function classSize(className, year = data.school.year) {
  if (apiMode) return data.analytics?.classSizes?.[`${year}|${className}`] || 0;
  return studentsInClass(className, year).filter((student) => student.active || year !== data.school.year).length;
}
function nextMatricule() {
  let sequence = data.students.length + 1;
  let matricule = `ELV-${new Date().getFullYear()}-${String(sequence).padStart(4, "0")}`;
  while (data.students.some((student) => student.matricule === matricule)) {
    sequence += 1;
    matricule = `ELV-${new Date().getFullYear()}-${String(sequence).padStart(4, "0")}`;
  }
  return matricule;
}
function totalPaid(studentId) {
  const computed = data.analytics?.studentFinancials?.[studentId]?.totalPaid;
  if (apiMode && Number.isFinite(computed)) return computed;
  return data.payments.filter((payment) => payment.studentId === studentId).reduce((sum, payment) => sum + Number(payment.amount), 0);
}
function outstanding(student) {
  const computed = data.analytics?.studentFinancials?.[student.id]?.outstanding;
  if (apiMode && Number.isFinite(computed)) return computed;
  return Math.max(0, Number(student.annualFee) - totalPaid(student.id));
}
function totalOutstanding() {
  if (apiMode && Number.isFinite(data.analytics?.dashboard?.outstanding)) return data.analytics.dashboard.outstanding;
  return data.students.filter((student) => student.active).reduce((sum, student) => sum + outstanding(student), 0);
}
function studentStatus(student) {
  const due = outstanding(student);
  if (!student.active) return ["Inactif", "inactive"];
  if (due === 0) return ["À jour", ""];
  return [totalPaid(student.id) ? "Partiel" : "À régler", "underpaid"];
}
function fmtDate(value) {
  if (!value) return "—";
  try { return frDate(value); } catch { return esc(value); }
}
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2700);
}
function pageHeading(title, subtitle, actions = "") {
  return `<div class="page-heading"><div><h1>${title}</h1><p>${subtitle}</p></div><div class="heading-actions">${actions}</div></div>`;
}
function btn(label, action, className = "button button-primary", icon = "") {
  return `<button type="button" class="${className}" data-action="${action}">${icon ? `<span aria-hidden="true">${icon}</span>` : ""}${label}</button>`;
}
function buttonForPermission(label, action, module, className, icon = "") {
  return can(module, action) ? btn(label, action, className, icon) : "";
}
function personCell(student) {
  return `<div class="person-cell"><span class="person-initials ${student.gender === "F" ? "female" : ""}">${esc(initials(`${student.firstName} ${student.lastName}`))}</span><span><strong>${esc(student.firstName)} ${esc(student.lastName)}</strong><small>${esc(student.guardian || "Responsable non renseigné")}</small></span></div>`;
}
function studentRows(students, withActions = true) {
  if (!students.length) return `<tr><td colspan="7"><div class="empty-state"><strong>Aucun élève trouvé</strong>Ajoutez un dossier ou modifiez votre recherche.</div></td></tr>`;
  return students.map((student) => {
    const [status, statusClass] = studentStatus(student);
    const actions = withActions ? `<td class="screen-only"><div class="table-actions">${can("dossiers", "read") ? `<button class="icon-button" data-action="view-student" data-id="${student.id}" aria-label="Ouvrir le dossier de ${esc(student.firstName)} ${esc(student.lastName)}">▣</button>` : ""}${can("students", "update") ? `<button class="icon-button" data-action="edit-student" data-id="${student.id}" aria-label="Modifier ${esc(student.firstName)} ${esc(student.lastName)}">✎</button>` : ""}${can("students", "delete") ? `<button class="icon-button" data-action="delete-student" data-id="${student.id}" aria-label="Supprimer ${esc(student.firstName)} ${esc(student.lastName)}">×</button>` : ""}</div></td>` : "";
    return `<tr><td>${personCell(student)}</td><td>${esc(student.className)}</td><td>${fmtDate(student.enrollmentDate)}</td><td>${money(student.annualFee)}</td><td>${money(outstanding(student))}</td><td><span class="status ${statusClass}">${status}</span></td>${actions}</tr>`;
  }).join("");
}
function studentsTable(students, { search = true, withActions = true, print = true } = {}) {
  const columns = `<th>Élève / responsable</th><th>Classe</th><th>Inscription</th><th>Frais annuels</th><th>Reste dû</th><th>Situation</th>${withActions ? '<th class="screen-only">Actions</th>' : ""}`;
  return `<section class="surface"><div class="surface-header"><div><h2>Liste des élèves</h2><p>${students.length} dossier${students.length === 1 ? "" : "s"} · Année scolaire ${esc(data.school.year)}</p></div>${print ? btn("Imprimer la liste", "print", "button button-secondary", "⎙") : ""}</div>${search ? `<div class="table-toolbar"><input class="search-control" type="search" placeholder="Rechercher un nom, une classe…" aria-label="Rechercher un élève" data-search="students"><div class="toolbar-group"><select class="select-control" data-filter="student-class" aria-label="Filtrer par classe"><option value="">Toutes les classes</option>${data.classes.map((item) => `<option value="${esc(item.name)}">${esc(item.name)}</option>`).join("")}</select><button class="button button-quiet" type="button" data-action="print">Imprimer</button></div></div>` : ""}<div class="table-wrap"><table><thead><tr>${columns}</tr></thead><tbody id="${search ? "studentRows" : ""}">${studentRows(students, withActions)}</tbody></table></div></section>`;
}
function paymentRows(payments, withActions = true) {
  if (!payments.length) return `<tr><td colspan="6"><div class="empty-state"><strong>Aucun paiement enregistré</strong>Les paiements apparaîtront ici après leur saisie.</div></td></tr>`;
  return payments.map((payment) => `<tr><td><strong>${esc(payment.reference || "—")}</strong></td><td>${esc(studentName(payment.studentId))}</td><td>${fmtDate(payment.date)}</td><td>${esc(payment.method)}</td><td><strong>${money(payment.amount)}</strong></td>${withActions ? `<td class="screen-only"><div class="table-actions">${can("payments", "update") ? `<button class="icon-button" data-action="edit-payment" data-id="${payment.id}" aria-label="Modifier le paiement">✎</button>` : ""}${can("payments", "delete") ? `<button class="icon-button" data-action="delete-payment" data-id="${payment.id}" aria-label="Supprimer le paiement">×</button>` : ""}</div></td>` : ""}</tr>`).join("");
}
function expensesRows(expenses) {
  if (!expenses.length) return `<tr><td colspan="6"><div class="empty-state"><strong>Aucune dépense enregistrée</strong>Ajoutez une écriture pour la suivre.</div></td></tr>`;
  return expenses.map((expense) => `<tr><td><strong>${esc(expense.label)}</strong><small class="muted"><br>${esc(expense.note || "")}</small></td><td>${esc(expense.category)}</td><td>${fmtDate(expense.date)}</td><td>${money(expense.amount)}</td><td><span class="status pending">Enregistrée</span></td><td class="screen-only"><div class="table-actions">${can("expenses", "update") ? `<button class="icon-button" data-action="edit-expense" data-id="${expense.id}" aria-label="Modifier la dépense">✎</button>` : ""}${can("expenses", "delete") ? `<button class="icon-button" data-action="delete-expense" data-id="${expense.id}" aria-label="Supprimer la dépense">×</button>` : ""}</div></td></tr>`).join("");
}
function svgChart() {
  const months = [];
  const now = new Date();
  for (let offset = 5; offset >= 0; offset--) {
    const date = new Date(now.getFullYear(), now.getMonth() - offset, 1);
    const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
    months.push({ key, label: new Intl.DateTimeFormat("fr-FR", { month: "short" }).format(date) });
  }
  const paid = months.map((month) => data.payments.filter((item) => item.date?.startsWith(month.key)).reduce((sum, item) => sum + Number(item.amount), 0));
  const spent = months.map((month) => data.expenses.filter((item) => item.date?.startsWith(month.key)).reduce((sum, item) => sum + Number(item.amount), 0));
  const max = Math.max(100000, ...paid, ...spent) * 1.15;
  const y = (amount) => 140 - (amount / max) * 120;
  const bars = months.map((month, index) => {
    const x = 54 + index * 77;
    const h1 = Math.max(2, 140 - y(paid[index]));
    const h2 = Math.max(2, 140 - y(spent[index]));
    return `<rect class="bar-paid" x="${x}" y="${y(paid[index])}" width="16" height="${h1}" rx="3"><title>${esc(month.label)} : recettes ${money(paid[index])}</title></rect><rect class="bar-expense" x="${x + 21}" y="${y(spent[index])}" width="16" height="${h2}" rx="3"><title>${esc(month.label)} : dépenses ${money(spent[index])}</title></rect><text x="${x + 16}" y="159" text-anchor="middle">${esc(month.label)}</text>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 500 165" role="img" aria-label="Graphique des recettes et dépenses sur six mois"><line class="grid-line" x1="36" y1="20" x2="495" y2="20"></line><line class="grid-line" x1="36" y1="60" x2="495" y2="60"></line><line class="grid-line" x1="36" y1="100" x2="495" y2="100"></line><line class="grid-line" x1="36" y1="140" x2="495" y2="140"></line><text x="0" y="23">${money(max).replace(/\s?FCFA$/, "")}</text><text x="0" y="63">${money(max * .66).replace(/\s?FCFA$/, "")}</text><text x="0" y="103">${money(max * .33).replace(/\s?FCFA$/, "")}</text><text x="0" y="143">0</text>${bars}</svg>`;
}
function dashboardMarkup() {
  const payments = [...data.payments].sort((a, b) => b.date.localeCompare(a.date));
  const recent = payments.slice(0, 4);
  const dashboard = data.analytics?.dashboard;
  const income = apiMode ? dashboard?.income || 0 : data.payments.reduce((sum, payment) => sum + Number(payment.amount), 0);
  const expenses = apiMode ? dashboard?.expenses || 0 : data.expenses.reduce((sum, item) => sum + Number(item.amount), 0);
  const activeStudents = apiMode ? dashboard?.activeStudents || 0 : data.students.filter((student) => student.active).length;
  const activeClasses = apiMode ? dashboard?.activeClasses || 0 : data.classes.filter((item) => item.active).length;
  const latestStudents = [...data.students].sort((a, b) => b.enrollmentDate.localeCompare(a.enrollmentDate)).slice(0, 4);
  const alerts = followUps().slice(0, 6);
  const activity = recent.length ? recent.map((payment) => `<div class="activity-row"><span class="activity-symbol">↙</span><span class="activity-copy"><strong>${esc(studentName(payment.studentId))}</strong><small>${esc(payment.method)} · ${fmtDate(payment.date)}</small></span><span class="activity-amount">+${money(payment.amount)}</span></div>`).join("") : `<div class="empty-state">Aucun mouvement pour le moment.</div>`;
  const classLines = data.classes.filter((item) => item.active).slice(0, 5).map((item) => `<div class="class-row"><span><strong>${esc(item.name)}</strong><small>${esc(item.teacher || "Enseignant à préciser")}</small></span><span class="class-count">${classSize(item.name)}<small>élèves</small></span></div>`).join("");
  return `${pageHeading("Tableau de bord", `${new Intl.DateTimeFormat("fr-FR", { weekday: "long", day: "numeric", month: "long" }).format(new Date())} · ${activeStudents} élèves actifs · ${activeClasses} classes actives`, `${btn("Nouvelle inscription", "new-student", "button button-primary", "+")}${btn("Enregistrer un paiement", "new-payment", "button button-secondary", "+")}`)}
    <section class="dashboard-ledger" aria-label="Situation financière et effectifs de l’établissement">
      <div class="ledger-head"><div><h2>Le registre, aujourd’hui</h2><p>Équation de caisse · Année scolaire ${esc(data.school.year)}</p></div><span class="ledger-file">${data.students.length} DOSSIERS · ${data.classes.length} CLASSES</span></div>
      <div class="ledger-equation">
        <div class="ledger-step"><span>RECETTES ENCAISSÉES</span><strong>${money(income)}</strong><small>${data.payments.length} paiements reçus</small></div>
        <span class="equation-mark" aria-hidden="true">−</span>
        <div class="ledger-step"><span>DÉPENSES ENREGISTRÉES</span><strong>${money(expenses)}</strong><small>${data.expenses.length} dépenses saisies</small></div>
        <span class="equation-mark" aria-hidden="true">=</span>
        <div class="ledger-step ledger-result"><span>SOLDE DE CAISSE</span><strong>${money(income - expenses)}</strong><small>Recettes moins dépenses</small></div>
      </div>
      <div class="ledger-credit"><span><strong>Crédits des familles</strong> · Solde restant à percevoir</span><strong>${money(totalOutstanding())}</strong></div>
    </section>
    <div class="dashboard-grid">
      <section class="surface"><div class="surface-header"><div><h2>Recettes & dépenses</h2><p>Écritures réelles · 6 derniers mois</p></div><span class="heading-date">${esc(data.school.year)}</span></div><div class="chart-wrap"><div class="chart-legend"><span class="legend-key"><i class="legend-dot"></i>Recettes</span><span class="legend-key"><i class="legend-dot orange"></i>Dépenses</span></div>${svgChart()}</div></section>
      <section class="surface"><div class="surface-header"><div><h2>Derniers paiements</h2><p>Mouvements les plus récents</p></div><button class="text-link" data-page-link="paiements">Tout voir →</button></div><div class="activity-list">${activity}</div></section>
    </div>
    <div class="dashboard-lower">
      <section class="surface"><div class="surface-header"><div><h2>Inscriptions récentes</h2><p>Derniers dossiers ajoutés à l’établissement</p></div><button class="text-link" data-page-link="eleves">Liste des élèves →</button></div><div class="table-wrap"><table><thead><tr><th>Élève / responsable</th><th>Classe</th><th>Date d’inscription</th><th>Situation</th></tr></thead><tbody>${latestStudents.length ? latestStudents.map((student) => { const [status, cls] = studentStatus(student); return `<tr><td>${personCell(student)}</td><td>${esc(student.className)}</td><td>${fmtDate(student.enrollmentDate)}</td><td><span class="status ${cls}">${status}</span></td></tr>`; }).join("") : `<tr><td colspan="4"><div class="empty-state">Aucun élève inscrit.</div></td></tr>`}</tbody></table></div></section>
      <section class="surface"><div class="surface-header"><div><h2>Classes actives</h2><p>Effectifs présents par classe</p></div><button class="text-link" data-page-link="classes">Gérer →</button></div>${classLines || `<div class="empty-state">Aucune classe active.</div>`}</section>
    </div><section class="surface follow-up-surface"><div class="surface-header"><div><h2>Suivis à traiter</h2><p>Rappels automatiques : impayés, dossiers incomplets et contrats proches de l’échéance.</p></div><span class="status ${alerts.length ? "pending" : ""}">${alerts.length} à suivre</span></div>${alerts.length ? `<ul class="follow-up-list">${alerts.map((item) => `<li><span class="follow-up-marker ${item.severity}"></span><span><strong>${esc(item.title)}</strong><small>${esc(item.detail)}</small></span><button class="text-link" data-action="${item.type === "teacher" ? "view-teacher" : "view-student"}" data-id="${item.id}">Ouvrir →</button></li>`).join("")}</ul>` : `<div class="empty-state">Tous les suivis sont à jour.</div>`}</section><p class="small-print muted">Les enregistrements de démonstration sont fictifs. Les montants sont calculés depuis les paiements et dépenses saisis.</p>`;
}
function studentsMarkup() {
  const actions = `${buttonForPermission("Inscrire un élève", "new-student", "students", "button button-primary", "+")}${btn("Imprimer la liste", "print", "button button-secondary", "⎙")}`;
  return `${pageHeading("Élèves", `${data.students.filter((student) => student.active).length} élèves actifs · ${data.students.length} dossiers`, actions)}${studentsTable(data.students)}`;
}
function yearlyRosterMarkup(schoolClass, year) {
  const roster = studentsInClass(schoolClass.name, year);
  const teacher = data.teachers.find((item) => item.id === schoolClass.teacherId);
  return `<details class="class-roster"><summary>Liste ${esc(year)} · ${roster.length} élève${roster.length === 1 ? "" : "s"}</summary><div class="table-wrap"><table><thead><tr><th>Élève</th><th>Matricule</th><th>Responsable</th><th class="screen-only">Liste</th></tr></thead><tbody>${roster.length ? roster.map((student, index) => `<tr><td>${index + 1}. ${esc(student.firstName)} ${esc(student.lastName)}</td><td>${esc(student.matricule || "—")}</td><td>${esc(student.guardian || "—")}</td><td class="screen-only"><button class="text-link" data-action="print-class-roster" data-id="${schoolClass.id}">Imprimer</button></td></tr>`).join("") : `<tr><td colspan="4">Aucun élève inscrit cette année.</td></tr>`}</tbody></table></div>${teacher && canRead("teachers") && canRead("students") ? `<p class="small-print muted">Professeur : ${esc(teacher.firstName)} ${esc(teacher.lastName)} · ${esc(teacher.phone || "contact manquant")}</p><button class="button button-secondary screen-only" data-action="whatsapp-roster" data-id="${schoolClass.id}">${data.school.whatsappMode === "link" ? "Préparer le lien WhatsApp" : "Tester la liste WhatsApp"}</button>` : ""}</details>`;
}
function annualPromotionMarkup(year) {
  const students = data.students.filter((student) => enrollmentForYear(student, year) && (year !== data.school.year || student.active));
  const closure = (data.school.academicClosures || []).find((item) => item.schoolYear === year);
  const threshold = Number(data.school.promotionThreshold ?? 10);
  const rows = students.map((student) => {
    const currentClass = data.classes.find((item) => item.name === enrollmentForYear(student, year)?.className);
    const level = academicLevel(currentClass || {});
    const average = annualAverageFor(student.id, year);
    const defaultDecision = annualDecisionFor(student, year);
    const decision = promotionDraft[student.id]?.decision || defaultDecision;
    const targets = classesAtNextLevels(level);
    const selectedTarget = promotionDraft[student.id]?.targetClassId || targets[0]?.id || "";
    const choices = average === null
      ? [{ value: "missing", label: "Notes incomplètes" }]
      : average < threshold
        ? [{ value: "repeat", label: "Redoublement · seuil non atteint" }]
        : TERMINAL_LEVELS.has(level)
          ? [{ value: "graduate", label: "Diplôme · fin de cycle" }]
          : [{ value: "promote", label: "Passage automatique au niveau supérieur" }];
    return `<tr><td>${personCell(student)}</td><td>${esc(enrollmentForYear(student, year)?.className || "—")}<small class="muted"><br>${esc(level || "Niveau à configurer")}</small></td><td>${average === null ? "—" : `${average.toFixed(2)} / 20`}</td><td><select class="select-control" data-year-decision="${student.id}" ${closure ? "disabled" : ""}>${choices.map((item) => `<option value="${item.value}" ${item.value === decision ? "selected" : ""}>${item.label}</option>`).join("")}</select>${decision === "promote" ? `<select class="select-control" data-year-target="${student.id}" ${closure || !targets.length ? "disabled" : ""}><option value="">Choisir la classe de destination</option>${targets.map((item) => `<option value="${item.id}" ${item.id === selectedTarget ? "selected" : ""}>${esc(item.name)} · ${esc(academicLevel(item))}</option>`).join("")}</select>${targets.length ? "" : `<small class="status pending">Créez une classe active du niveau ${esc((NEXT_LEVELS[level] || []).join(" / ") || "suivant")}.</small>`}` : ""}</td></tr>`;
  }).join("");
  const allReady = students.every((student) => {
    const decision = promotionDraft[student.id]?.decision || annualDecisionFor(student, year);
    if (decision === "missing") return false;
    if (decision !== "promote") return true;
    const currentClass = data.classes.find((item) => item.name === enrollmentForYear(student, year)?.className);
    return classesAtNextLevels(academicLevel(currentClass || {})).length > 0;
  });
  const canClose = can("students", "update") && apiMode && year === data.school.year && !closure && students.length > 0 && allReady;
  return `<section class="surface annual-board"><div class="surface-header"><div><h2>Bilan annuel & passage</h2><p>Moyenne générale pondérée par les coefficients · seuil configurable : ${threshold.toFixed(1)} / 20.</p></div>${closure ? `<span class="status">Année clôturée · ${esc(closure.nextSchoolYear)}</span>` : year === data.school.year ? `<button class="button button-primary" data-action="close-academic-year" ${canClose ? "" : "disabled"}>Valider le bilan et passer à l’année suivante</button>` : `<span class="status pending">Consultation historique</span>`}</div>${students.length ? `<div class="table-wrap"><table><thead><tr><th>Élève</th><th>Classe</th><th>Moyenne générale</th><th>Décision / destination</th></tr></thead><tbody>${rows}</tbody></table></div>` : `<div class="empty-state">Aucun élève actif inscrit pour ${esc(year)}.</div>`}<p class="small-print muted">Les passages sont calculés à partir des notes de l’année choisie. Une classe active de destination doit exister pour chaque promotion. La validation conserve le bilan et l’inscription de l’année clôturée, puis actualise l’année active; la 12e est le dernier niveau du cycle.${apiMode ? "" : " La clôture sécurisée est disponible via le serveur FastAPI."}</p></section>`;
}
function classesMarkup() {
  selectedAcademicYear ||= data.school.year;
  const year = yearsInData().includes(selectedAcademicYear) ? selectedAcademicYear : data.school.year;
  const activeClasses = data.classes.filter((item) => item.active);
  const cards = activeClasses.length ? activeClasses.map((item) => `<article class="class-tile"><h3>${esc(item.name)}</h3><p>${esc(item.level || "Niveau non précisé")} · ${esc(academicLevel(item) || "Niveau à configurer")}${item.teacher ? ` · ${esc(item.teacher)}` : ""}</p><div class="class-tile-bottom"><strong>${classSize(item.name, year)} / ${Number(item.capacity) || "—"} élèves · ${esc(year)}</strong><span class="class-fee">${money(item.annualFee)} / an</span></div><div class="table-actions" style="margin-top:12px">${can("classes", "update") ? `<button class="button button-quiet" data-action="edit-class" data-id="${item.id}">Modifier</button>` : ""}${can("classes", "delete") ? `<button class="button button-quiet" data-action="delete-class" data-id="${item.id}">Supprimer</button>` : ""}</div>${yearlyRosterMarkup(item, year)}</article>`).join("") : `<div class="surface empty-state"><strong>Aucune classe</strong>Créez les classes de l’établissement pour inscrire vos élèves.</div>`;
  return `${pageHeading("Classes", `${activeClasses.length} classes actives · ${data.students.filter((student) => student.active && enrollmentForYear(student, year)).length} élèves inscrits en ${esc(year)}`, `${buttonForPermission("Créer une classe", "new-class", "classes", "button button-primary", "+")}${btn("Imprimer les listes", "print", "button button-secondary", "⎙")}`)}<div class="inline-filters year-picker"><label class="field-label" for="classAcademicYear">Année scolaire</label><select class="select-control" id="classAcademicYear">${yearsInData().map((item) => `<option value="${esc(item)}" ${item === year ? "selected" : ""}>${esc(item)}</option>`).join("")}</select><span class="small-print muted">Listes recalculées selon les inscriptions conservées pour chaque année.</span></div><div class="class-grid">${cards}</div>${annualPromotionMarkup(year)}`;
}
function paymentsMarkup() {
  const total = apiMode ? data.analytics?.dashboard?.income || 0 : data.payments.reduce((sum, payment) => sum + Number(payment.amount), 0);
  const sorted = [...data.payments].sort((a, b) => b.date.localeCompare(a.date));
  return `${pageHeading("Paiements", `Suivi des frais scolaires · ${data.payments.length} transactions enregistrées`, `${buttonForPermission("Nouveau paiement", "new-payment", "payments", "button button-primary", "+")}${btn("Imprimer le journal", "print", "button button-secondary", "⎙")}`)}
    <section class="metric-grid"><div class="metric"><div class="metric-top"><span class="metric-label">ENCAISSEMENTS CUMULÉS</span><span class="metric-mark green">↙</span></div><div class="metric-value">${money(total)}</div><div class="metric-detail">Toutes périodes confondues</div></div><div class="metric"><div class="metric-top"><span class="metric-label">RESTE À ENCAISSER</span><span class="metric-mark orange">◷</span></div><div class="metric-value">${money(totalOutstanding())}</div><div class="metric-detail">${data.students.filter((student) => outstanding(student) > 0).length} élèves avec un solde</div></div><div class="metric"><div class="metric-top"><span class="metric-label">MODES DE PAIEMENT</span><span class="metric-mark">⇄</span></div><div class="metric-value">${new Set(data.payments.map((payment) => payment.method)).size}</div><div class="metric-detail">${METHODS.length} modes disponibles</div></div><div class="metric"><div class="metric-top"><span class="metric-label">REÇU MOYEN</span><span class="metric-mark green">◎</span></div><div class="metric-value">${money(data.payments.length ? total / data.payments.length : 0)}</div><div class="metric-detail">Sur l’ensemble des paiements</div></div></section>
    <section class="surface"><div class="surface-header"><div><h2>Journal des paiements</h2><p>Chaque versement est associé à un élève et à son moyen de paiement.</p></div></div><div class="table-wrap"><table><thead><tr><th>Référence</th><th>Élève</th><th>Date</th><th>Moyen de paiement</th><th>Montant</th><th>Actions</th></tr></thead><tbody>${paymentRows(sorted)}</tbody></table></div></section>`;
}
function expensesMarkup() {
  const dashboard = data.analytics?.dashboard;
  const total = apiMode ? dashboard?.expenses || 0 : data.expenses.reduce((sum, item) => sum + Number(item.amount), 0);
  const income = apiMode ? dashboard?.income || 0 : data.payments.reduce((sum, payment) => sum + Number(payment.amount), 0);
  const sorted = [...data.expenses].sort((a, b) => b.date.localeCompare(a.date));
  return `${pageHeading("Dépenses", `Journal des sorties de caisse · ${data.expenses.length} dépenses`, `${buttonForPermission("Nouvelle dépense", "new-expense", "expenses", "button button-primary", "+")}${btn("Imprimer le journal", "print", "button button-secondary", "⎙")}`)}
    <section class="metric-grid"><div class="metric"><div class="metric-top"><span class="metric-label">TOTAL DES DÉPENSES</span><span class="metric-mark red">↗</span></div><div class="metric-value">${money(total)}</div><div class="metric-detail">${data.expenses.length} écritures</div></div><div class="metric"><div class="metric-top"><span class="metric-label">RECETTES NETTES</span><span class="metric-mark green">↔</span></div><div class="metric-value">${money(income - total)}</div><div class="metric-detail">Recettes moins dépenses</div></div><div class="metric"><div class="metric-top"><span class="metric-label">ÉLÈVES À JOUR</span><span class="metric-mark green">✓</span></div><div class="metric-value">${data.students.filter((student) => student.active && outstanding(student) === 0).length}</div><div class="metric-detail">Frais annuels réglés</div></div><div class="metric"><div class="metric-top"><span class="metric-label">CRÉDITS SCOLAIRES</span><span class="metric-mark orange">◷</span></div><div class="metric-value">${money(totalOutstanding())}</div><div class="metric-detail">Montant dû par les familles</div></div></section>
    <section class="surface"><div class="surface-header"><div><h2>Journal des dépenses</h2><p>Montants en FCFA · Les totaux sont calculés automatiquement.</p></div></div><div class="table-wrap"><table><thead><tr><th>Libellé</th><th>Catégorie</th><th>Date</th><th>Montant</th><th>Statut</th><th>Actions</th></tr></thead><tbody>${expensesRows(sorted)}</tbody></table></div></section>`;
}
function accountingMarkup() {
  const matchesPeriod = (entry) => (!selectedAccountingPeriod.start || entry.date >= selectedAccountingPeriod.start)
    && (!selectedAccountingPeriod.end || entry.date <= selectedAccountingPeriod.end);
  const receipts = data.payments.filter(matchesPeriod);
  const expenses = data.expenses.filter(matchesPeriod);
  const income = apiMode
    ? data.analytics?.period?.income || 0
    : receipts.reduce((sum, item) => sum + Number(item.amount), 0);
  const spending = apiMode
    ? data.analytics?.period?.expenses || 0
    : expenses.reduce((sum, item) => sum + Number(item.amount), 0);
  const entries = [
    ...receipts.map((item) => ({ date: item.date, type: "Recette", label: `Paiement · ${studentName(item.studentId)}`, method: item.method, reference: item.reference, amount: Number(item.amount) })),
    ...expenses.map((item) => ({ date: item.date, type: "Dépense", label: item.label, method: item.category, reference: item.note, amount: -Number(item.amount) })),
  ].sort((a, b) => b.date.localeCompare(a.date));
  const periodLabel = selectedAccountingPeriod.start || selectedAccountingPeriod.end
    ? `Du ${fmtDate(selectedAccountingPeriod.start)} au ${fmtDate(selectedAccountingPeriod.end)}`
    : "Toutes périodes";
  return `${pageHeading("Comptabilité & rapports", "Vue consolidée des recettes, dépenses, crédits et moyens de paiement.", `${apiMode && can("accounting", "read") ? btn("Télécharger le rapport PDF Python", "download-financial-pdf", "button button-secondary", "↓") : ""}${btn("Imprimer le rapport", "print", "button button-primary", "⎙")}`)}
    <div class="inline-filters accounting-period screen-only"><label class="field"><span class="field-label">Du</span><input id="accountingFrom" class="field-control" type="date" value="${esc(selectedAccountingPeriod.start)}"></label><label class="field"><span class="field-label">Au</span><input id="accountingTo" class="field-control" type="date" value="${esc(selectedAccountingPeriod.end)}"></label><span class="small-print muted">La période filtre les mouvements; les crédits restent le solde actuel des familles.</span><button class="button button-quiet" type="button" data-action="clear-accounting-period">Effacer les dates</button></div>
    <p class="print-only accounting-print-period">Période du rapport : ${esc(periodLabel)}</p>
    <section class="metric-grid"><div class="metric"><div class="metric-top"><span class="metric-label">TOTAL DES RECETTES</span><span class="metric-mark green">↙</span></div><div class="metric-value">${money(income)}</div><div class="metric-detail">${receipts.length} versements · ${esc(periodLabel)}</div></div><div class="metric"><div class="metric-top"><span class="metric-label">TOTAL DES DÉPENSES</span><span class="metric-mark red">↗</span></div><div class="metric-value">${money(spending)}</div><div class="metric-detail">${expenses.length} dépenses · ${esc(periodLabel)}</div></div><div class="metric"><div class="metric-top"><span class="metric-label">SOLDE DE CAISSE</span><span class="metric-mark green">＝</span></div><div class="metric-value">${money(income - spending)}</div><div class="metric-detail">Mouvements de la période</div></div><div class="metric"><div class="metric-top"><span class="metric-label">CRÉDIT FAMILLES</span><span class="metric-mark orange">◷</span></div><div class="metric-value">${money(totalOutstanding())}</div><div class="metric-detail">${data.students.filter((student) => student.active && outstanding(student) > 0).length} dossiers avec un solde dû · situation actuelle</div></div></section>
    <section class="surface"><div class="surface-header"><div><h2>Grand livre</h2><p>Écritures datées, liées aux reçus et aux élèves · ${esc(periodLabel)}.</p></div><span class="status">${entries.length} écritures</span></div><div class="table-wrap"><table><thead><tr><th>Date</th><th>Type</th><th>Libellé</th><th>Moyen / catégorie</th><th>Référence</th><th>Montant</th></tr></thead><tbody>${entries.length ? entries.map((entry) => `<tr><td>${fmtDate(entry.date)}</td><td><span class="status ${entry.type === "Recette" ? "" : "pending"}">${entry.type}</span></td><td>${esc(entry.label)}</td><td>${esc(entry.method || "—")}</td><td>${esc(entry.reference || "—")}</td><td><strong>${money(entry.amount)}</strong></td></tr>`).join("") : `<tr><td colspan="6">Aucun mouvement sur cette période.</td></tr>`}</tbody></table></div><div class="ledger-credit"><span><strong>Solde net · ${esc(periodLabel)}</strong> · Recettes − dépenses</span><strong>${money(income - spending)}</strong></div></section>
    <p class="small-print muted">Le grand livre est calculé automatiquement depuis les paiements et dépenses. Modifiez une écriture depuis son module d’origine.</p>`;
}
function gradesFor(studentId, year = selectedEvaluationYear || data.school.year, term = selectedGradeTerm) {
  return data.grades.filter((grade) => grade.studentId === studentId
    && grade.term === term
    && (grade.schoolYear || data.assessments.find((item) => item.id === grade.assessmentId)?.schoolYear || data.school.year) === year);
}
function marksFor(studentId) { return gradesFor(studentId).filter((grade) => Number.isFinite(grade.score)); }
function averageFor(studentId) {
  const key = `${studentId}|${selectedEvaluationYear || data.school.year}|${selectedGradeTerm}`;
  const computed = data.analytics?.averages?.[key]?.average;
  if (apiMode) return Number.isFinite(computed) ? computed : null;
  const marks = marksFor(studentId);
  const weight = marks.reduce((sum, grade) => sum + Number(grade.coefficient || 1), 0);
  return weight ? marks.reduce((sum, grade) => sum + Number(grade.score) * Number(grade.coefficient || 1), 0) / weight : null;
}
function gradeComment(average) {
  if (average === null) return "Aucune note n’a encore été saisie.";
  if (average >= 16) return "Excellent travail. Continuez avec la même régularité.";
  if (average >= 14) return "Très bon trimestre. Les efforts portent leurs fruits.";
  if (average >= 12) return "Bon travail. Il faut poursuivre les efforts.";
  if (average >= 10) return "Résultats satisfaisants. Un travail régulier aidera à progresser.";
  return "Des efforts supplémentaires sont nécessaires. Un accompagnement est recommandé.";
}
function evaluationHistoryMarkup(student) {
  const years = [...new Set([
    ...(student.enrollmentHistory || []).map((item) => item.schoolYear),
    ...data.grades.filter((grade) => grade.studentId === student.id).map((grade) => grade.schoolYear || data.school.year),
  ].filter(Boolean))].sort((a, b) => a.localeCompare(b));
  const yearSections = years.map((year) => {
    const grades = data.grades.filter((grade) => grade.studentId === student.id
      && (grade.schoolYear || data.assessments.find((item) => item.id === grade.assessmentId)?.schoolYear || data.school.year) === year);
    const scored = grades.filter((grade) => Number.isFinite(grade.score));
    const weight = scored.reduce((sum, grade) => sum + Number(grade.coefficient || 1), 0);
    const average = apiMode ? annualAverageFor(student.id, year) : weight ? scored.reduce((sum, grade) => sum + Number(grade.score) * Number(grade.coefficient || 1), 0) / weight : null;
    const enrollment = enrollmentForYear(student, year);
    const previousYear = years[years.indexOf(year) - 1];
    const previousAverage = previousYear ? annualAverageFor(student.id, previousYear) : null;
    const subjects = [...new Set(grades.map((grade) => grade.subject))];
    const subjectSummary = subjects.map((subject) => {
      const marks = scored.filter((grade) => grade.subject === subject);
      const subjectWeight = marks.reduce((sum, grade) => sum + Number(grade.coefficient || 1), 0);
      const computedSubjectAverage = data.analytics?.subjectAverages?.[`${student.id}|${year}|annual|${subject}`]?.average;
      const subjectAverage = apiMode
        ? Number.isFinite(computedSubjectAverage) ? computedSubjectAverage : null
        : subjectWeight ? marks.reduce((sum, grade) => sum + Number(grade.score) * Number(grade.coefficient || 1), 0) / subjectWeight : null;
      return `<tr><td>${esc(subject)}</td><td>${marks.length}</td><td>${subjectAverage === null ? "—" : `${subjectAverage.toFixed(2)} / 20`}</td></tr>`;
    }).join("");
    return `<section class="evaluation-year"><h3>${esc(year)} · ${esc(enrollment?.className || "Classe historique inconnue")}</h3><p class="evaluation-progress">Moyenne générale : <strong>${average === null ? "—" : `${average.toFixed(2)} / 20`}</strong> · Progression : <strong>${average === null || previousAverage === null ? "Non calculable" : `${average - previousAverage >= 0 ? "+" : ""}${(average - previousAverage).toFixed(2)} points`}</strong></p><table class="report-table"><thead><tr><th>Date</th><th>Trimestre</th><th>Matière</th><th>Évaluation</th><th>Note / 20</th><th>Coef.</th><th>Points</th></tr></thead><tbody>${grades.length ? grades.sort((a, b) => (a.date || "").localeCompare(b.date || "")).map((grade) => `<tr><td>${fmtDate(grade.date || data.assessments.find((item) => item.id === grade.assessmentId)?.date)}</td><td>${esc(grade.term || "—")}</td><td>${esc(grade.subject || "—")}</td><td>${esc(grade.type || data.assessments.find((item) => item.id === grade.assessmentId)?.title || "Évaluation")}</td><td>${Number.isFinite(grade.score) ? Number(grade.score).toFixed(2) : "—"}</td><td>${Number(grade.coefficient) || 1}</td><td>${Number.isFinite(grade.score) ? (Number(grade.score) * Number(grade.coefficient || 1)).toFixed(2) : "—"}</td></tr>`).join("") : `<tr><td colspan="7">Aucune note enregistrée pour cette année.</td></tr>`}</tbody></table><table class="evaluation-subjects"><thead><tr><th>Bilan par matière</th><th>Évaluations notées</th><th>Moyenne pondérée</th></tr></thead><tbody>${subjectSummary || `<tr><td colspan="3">Aucune matière évaluée.</td></tr>`}</tbody></table><p class="report-comment"><strong>Appréciation :</strong> ${esc(gradeComment(average))}</p></section>`;
  }).join("");
  return `<article class="report-sheet evaluation-history-sheet"><div class="report-school"><div class="document-identity">${data.school.logo ? `<img class="document-logo" src="${esc(data.school.logo)}" alt="Logo de l’établissement">` : `<span class="document-logo-fallback">${esc(initials(data.school.shortName || data.school.name))}</span>`}${MALI_FLAG}</div><div><h2>${esc(data.school.name)}</h2><p>${esc(data.school.address)} · ${esc(data.school.phone)}</p></div></div><h2 class="report-title">DOSSIER D’ÉVALUATION & PROGRESSION</h2><div class="report-info"><div>Élève : <strong>${esc(student.firstName)} ${esc(student.lastName)}</strong></div><div>Matricule : <strong>${esc(student.matricule || student.id.toUpperCase())}</strong></div><div>Années suivies : <strong>${years.length}</strong></div><div>Situation : <strong>${esc(studentStatus(student)[0])}</strong></div></div>${yearSections || `<p>Aucune inscription annuelle ou évaluation n’est enregistrée.</p>`}</article>`;
}
function gradeRows(student) {
  const grades = gradesFor(student.id);
  if (!grades.length) return `<tr><td colspan="5"><div class="empty-state">Aucune matière notée. Ajoutez une note pour commencer le bulletin.</div></td></tr>`;
  return grades.map((grade) => `<tr><td>${esc(grade.subject)}<small class="muted"><br>${esc(grade.type || "Évaluation")}</small></td><td><input class="grades-input" type="number" min="0" max="20" step="0.25" value="${Number.isFinite(grade.score) ? Number(grade.score) : ""}" data-grade-score="${grade.id}" aria-label="Note de ${esc(grade.subject)} pour ${esc(student.firstName)}" ${can("grades", "update") && selectedEvaluationYear === data.school.year ? "" : "disabled"}></td><td>${Number(grade.coefficient) || 1}</td><td class="grade-average">${Number.isFinite(grade.score) ? (Number(grade.score) * Number(grade.coefficient || 1)).toFixed(2) : "—"}</td><td>${can("grades", "delete") && selectedEvaluationYear === data.school.year ? `<button class="icon-button" data-action="delete-grade" data-id="${grade.id}" aria-label="Supprimer la note de ${esc(grade.subject)}">×</button>` : ""}</td></tr>`).join("");
}
function gradeEditor(student) {
  const average = averageFor(student.id);
  return `<section class="surface"><div class="surface-header"><div>${personCell(student)}</div><div class="inline-filters"><span class="status">Moyenne : ${average === null ? "—" : `${average.toFixed(2)} / 20`}</span>${can("grades", "create") ? `<button class="button button-secondary" data-action="new-grade" data-id="${student.id}">Ajouter une matière +</button>` : ""}<button class="button button-primary" data-action="print-report" data-id="${student.id}">Imprimer le bulletin</button></div></div><div class="table-wrap"><table><thead><tr><th>Matière</th><th>Note / 20</th><th>Coefficient</th><th>Points</th><th></th></tr></thead><tbody>${gradeRows(student)}</tbody></table></div></section>`;
}
function assessmentEditor(assessment) {
  if (!assessment) return "";
  const students = data.students.filter((student) => enrollmentForYear(student, assessment.schoolYear || data.school.year)?.className === assessment.className);
  const entries = students.map((student) => {
    const grade = data.grades.find((item) => item.assessmentId === assessment.id && item.studentId === student.id);
    return `<tr><td>${personCell(student)}</td><td>${grade ? `<input class="grades-input" type="number" min="0" max="20" step="0.25" value="${Number.isFinite(grade.score) ? grade.score : ""}" data-assessment-score="${grade.id}" aria-label="Note de ${esc(student.firstName)} ${esc(student.lastName)}" ${can("grades", "update") && assessment.schoolYear === data.school.year ? "" : "disabled"}>` : "—"}</td><td>${grade && Number.isFinite(grade.score) ? `${(grade.score * Number(grade.coefficient || 1)).toFixed(2)} pts` : "—"}</td></tr>`;
  }).join("");
  const scored = students.map((student) => data.grades.find((item) => item.assessmentId === assessment.id && item.studentId === student.id)?.score).filter(Number.isFinite);
  const average = scored.length ? scored.reduce((sum, score) => sum + score, 0) / scored.length : null;
  return `<section class="surface assessment-editor"><div class="surface-header"><div><h2>${esc(assessment.title)}</h2><p>${esc(assessment.className)} · ${esc(assessment.subject)} · ${esc(assessment.type)} · ${fmtDate(assessment.date)}</p></div><div class="inline-filters"><span class="status">${scored.length} / ${students.length} notes · Moyenne ${average === null ? "—" : average.toFixed(2)}</span>${can("grades", "delete") ? `<button class="button button-quiet" data-action="delete-assessment" data-id="${assessment.id}">Supprimer l’évaluation</button>` : ""}</div></div><div class="table-wrap"><table><thead><tr><th>Élève</th><th>Note / 20</th><th>Points</th></tr></thead><tbody>${entries || `<tr><td colspan="3">Aucun élève actif dans cette classe.</td></tr>`}</tbody></table></div></section>`;
}
function gradesMarkup() {
  selectedEvaluationYear ||= data.school.year;
  if (!yearsInData().includes(selectedEvaluationYear)) selectedEvaluationYear = data.school.year;
  const activeStudents = data.students.filter((student) => student.active);
  const student = activeStudents.find((item) => item.id === selectedGradeStudentId) || activeStudents[0];
  selectedGradeStudentId = student?.id || null;
  const assessments = data.assessments.filter((item) => item.schoolYear === selectedEvaluationYear);
  const assessment = assessments.find((item) => item.id === selectedAssessmentId);
  const assessmentRows = assessments.length ? [...assessments].sort((a, b) => b.date.localeCompare(a.date)).map((item) => `<tr><td><strong>${esc(item.title)}</strong><small class="muted"><br>${esc(item.type)} · ${esc(item.subject)}</small></td><td>${esc(item.className)}</td><td>${esc(item.term)}</td><td>${fmtDate(item.date)}</td><td><button class="text-link" data-action="select-assessment" data-id="${item.id}">Saisir les notes →</button></td></tr>`).join("") : `<tr><td colspan="5"><div class="empty-state">Aucune composition créée pour cette année scolaire.</div></td></tr>`;
  return `${pageHeading("Notes & bulletins", "Saisissez les notes par élève et calculez automatiquement les moyennes.", `${student ? `<button class="button button-secondary" data-action="print-class-list">Imprimer la liste de classe</button><button class="button button-secondary" data-action="print-evaluation-history" data-id="${student.id}">Imprimer le suivi annuel</button>` : ""}`)}
    <section class="surface"><div class="surface-header"><div><h2>Carnet de notes</h2><p>Les notes sont sur 20. La moyenne est pondérée par matière et coefficient; les années clôturées restent consultables sans modification.</p></div><div class="inline-filters">${student ? `<label class="field-label" for="gradeStudent">Élève</label><select id="gradeStudent" class="select-control">${activeStudents.map((item) => `<option value="${item.id}" ${item.id === student.id ? "selected" : ""}>${esc(item.firstName)} ${esc(item.lastName)} · ${esc(item.className)}</option>`).join("")}</select>` : ""}<select id="evaluationYear" class="select-control" aria-label="Année scolaire">${yearsInData().map((year) => `<option value="${esc(year)}" ${year === selectedEvaluationYear ? "selected" : ""}>${esc(year)}</option>`).join("")}</select><select id="gradeTerm" class="select-control" aria-label="Trimestre">${["1er trimestre", "2e trimestre", "3e trimestre"].map((term) => `<option ${term === selectedGradeTerm ? "selected" : ""}>${term}</option>`).join("")}</select></div></div>${student ? gradeEditor(student) : `<div class="empty-state"><strong>Aucun élève inscrit</strong>Inscrivez d’abord un élève avant de saisir ses notes.</div>`}</section>
    <div class="surface" style="margin-top:16px"><div class="surface-header"><div><h2>Notes de classe · devoirs & compositions</h2><p>Créez une évaluation puis saisissez toutes les notes de la classe dans la même grille.</p></div>${can("grades", "create") && selectedEvaluationYear === data.school.year ? `<button class="button button-primary" data-action="new-assessment">Nouvelle composition</button>` : ""}</div><div class="table-wrap"><table><thead><tr><th>Évaluation</th><th>Classe</th><th>Trimestre</th><th>Date</th><th>Notes</th></tr></thead><tbody>${assessmentRows}</tbody></table></div>${assessmentEditor(assessment)}</div>
    <div class="surface" style="margin-top:16px"><div class="surface-header"><div><h2>Bulletins par classe</h2><p>Chaque bulletin utilise les notes enregistrées pour cet élève et ce trimestre.</p></div></div><div class="table-wrap"><table><thead><tr><th>Élève</th><th>Classe</th><th>Moyenne</th><th>Appréciation</th><th>Document</th></tr></thead><tbody>${activeStudents.length ? activeStudents.map((item) => { const average = averageFor(item.id); return `<tr><td>${personCell(item)}</td><td>${esc(item.className)}</td><td>${average === null ? "—" : `${average.toFixed(2)} / 20`}</td><td>${esc(gradeComment(average))}</td><td><button class="text-link" data-action="print-report" data-id="${item.id}">Imprimer →</button></td></tr>`; }).join("") : `<tr><td colspan="5"><div class="empty-state">Aucun dossier élève.</div></td></tr>`}</tbody></table></div></div>`;
}
function reportMarkup(student) {
  const marks = marksFor(student.id);
  const average = averageFor(student.id);
  return `<article class="report-sheet"><div class="report-school"><div class="document-identity">${data.school.logo ? `<img class="document-logo" src="${esc(data.school.logo)}" alt="Logo de l’établissement">` : `<span class="document-logo-fallback">${esc(initials(data.school.shortName || data.school.name))}</span>`}${MALI_FLAG}</div><div><h2>${esc(data.school.name)}</h2><p>${esc(data.school.address)} · ${esc(data.school.phone)}</p></div></div><h3 class="report-title">BULLETIN DE NOTES · ${esc(selectedGradeTerm)}</h3><div class="report-info"><div><span>Élève :</span> <strong>${esc(student.firstName)} ${esc(student.lastName)}</strong></div><div><span>Classe :</span> <strong>${esc(enrollmentForYear(student, selectedEvaluationYear)?.className || student.className)}</strong></div><div><span>Année scolaire :</span> <strong>${esc(selectedEvaluationYear)}</strong></div><div><span>Matricule :</span> <strong>${esc(student.matricule || student.id.toUpperCase())}</strong></div></div><table class="report-table"><thead><tr><th>Matière</th><th>Note / 20</th><th>Coefficient</th><th>Points</th></tr></thead><tbody>${marks.length ? marks.map((grade) => `<tr><td>${esc(grade.subject)}</td><td>${Number(grade.score).toFixed(2)}</td><td>${Number(grade.coefficient) || 1}</td><td>${(Number(grade.score) * Number(grade.coefficient || 1)).toFixed(2)}</td></tr>`).join("") : `<tr><td colspan="4">Aucune note saisie pour le moment.</td></tr>`}${canRead("discipline") ? `<tr class="conduct-grade"><td>Conduite · points retirés uniquement après validation</td><td>${conductBalance(student.id)} / ${conductCapital()}</td><td>—</td><td>Retraits validés : ${Math.max(0, conductCapital() - conductBalance(student.id))} pts</td></tr>` : ""}</tbody></table><div class="report-average"><span>Moyenne générale</span><strong>${average === null ? "—" : `${average.toFixed(2)} / 20`}</strong></div><p class="report-comment"><strong>Appréciation :</strong> ${esc(gradeComment(average))}</p>${canRead("discipline") ? `<p class="report-comment"><strong>Conduite :</strong> ${conductBalance(student.id)} / ${conductCapital()} points. Le solde est calculé depuis les retraits validés.</p>` : ""}<div class="report-signatures"><span>Signature du responsable</span><span>Cachet et signature de la direction</span></div></article>`;
}
function cardsMarkup() {
  const template = data.school.cardTemplate;
  const students = data.students.filter((student) => student.active);
  const cards = students.map((student) => {
    const details = [
      `<span>CLASSE<strong>${esc(student.className)}</strong></span>`,
      template.showBirthDate ? `<span>NÉ(E) LE<strong>${fmtDate(student.birthDate)}</strong></span>` : "",
      template.showGuardian ? `<span>RESPONSABLE<strong>${esc(student.guardian || "—")}</strong></span>` : "",
      template.showFamilyPhone ? `<span>CONTACT FAMILLE<strong>${esc(student.phone || "—")}</strong></span>` : "",
    ].filter(Boolean).join("");
    return `<article class="student-card ${template.layout === "vertical" ? "student-card--vertical" : ""}"><div class="student-card-main"><div class="student-card-head"><div class="document-identity">${data.school.logo ? `<img class="document-logo" src="${esc(data.school.logo)}" alt="Logo de l’établissement">` : `<span class="document-logo-fallback">${esc(initials(data.school.shortName || data.school.name))}</span>`}${MALI_FLAG}</div><div><strong>${esc(data.school.name)}</strong><small>CARTE SCOLAIRE · ${esc(data.school.year)}</small></div></div><div class="student-card-info"><span>ÉLÈVE<strong>${esc(student.firstName)} ${esc(student.lastName)}</strong></span>${details}${template.showMatricule ? `<span>MATRICULE<strong>${esc(student.matricule || student.id.slice(0, 12).toUpperCase())}</strong></span>` : ""}</div></div><aside class="student-card-aside">${template.showPhoto && student.photo ? `<img class="card-photo" src="${esc(student.photo)}" alt="Photo de ${esc(student.firstName)} ${esc(student.lastName)}">` : `<div class="card-stamp" aria-label="Aucune photo, initiales de ${esc(student.firstName)} ${esc(student.lastName)}">${esc(initials(`${student.firstName} ${student.lastName}`))}</div>`}</aside></article>`;
  }).join("");
  return `${pageHeading("Cartes scolaires", `Modèle personnalisable · ${students.length} cartes nominatives prêtes à imprimer.`, `${btn("Imprimer les cartes", "print", "button button-primary", "⎙")}`)}<div class="class-grid">${cards || `<div class="surface empty-state"><strong>Aucun élève inscrit</strong>Les cartes apparaîtront après les premières inscriptions.</div>`}</div><p class="small-print muted">Le drapeau du Mali (SVG local) et le logo configuré figurent sur les cartes et bulletins; le modèle et les champs facultatifs se règlent dans Paramètres.</p>`;
}
function permissionsMarkup(user) {
  const operator = currentUser();
  return `<section class="surface"><div class="surface-header"><div><h2>Droits de ${esc(user.username)}</h2><p>Choisissez les actions autorisées pour chaque espace. Les changements sont enregistrés immédiatement.</p></div><span class="status ${user.active ? "" : "inactive"}">${user.active ? "Compte actif" : "Compte désactivé"}</span></div><p class="permission-notice">Les journaux d’audit sont append-only. L’activation ou la désactivation des comptes est réservée aux comptes disposant du droit dédié.</p><div class="table-wrap"><table class="permission-table"><thead><tr><th>Rubrique</th><th>Consulter</th><th>Créer</th><th>Modifier</th><th>Supprimer</th><th>Activer un compte</th></tr></thead><tbody>${Object.entries(MODULES).map(([key, label]) => `<tr><td>${esc(label)}</td>${ACTIONS.map((action) => { const restricted = user.role === "proviseur" || user.id === currentUserId || !can("accounts", "update") || (operator?.role === "admin" && !operator.permissions?.[key]?.[action]); return `<td><input type="checkbox" data-permission="${key}.${action}" ${user.permissions?.[key]?.[action] ? "checked" : ""} ${restricted ? "disabled" : ""} aria-label="${esc(label)} : ${action === "read" ? "consulter" : action === "create" ? "créer" : action === "update" ? "modifier" : "supprimer"}"></td>`; }).join("")}<td>${key === "accounts" ? `<input type="checkbox" data-permission="accounts.activate" ${user.permissions?.accounts?.activate ? "checked" : ""} ${user.role === "proviseur" || user.id === currentUserId || !can("accounts", "update") || (operator?.role === "admin" && !operator.permissions?.accounts?.activate) ? "disabled" : ""} aria-label="Activer ou désactiver un compte">` : "—"}</td></tr>`).join("")}</tbody></table></div><p class="small-print muted">La portée de classe ou de cycle est appliquée par le serveur. Les comptes Direction restent confinés à leur cycle d’affectation.</p></section>`;
}
function accountsMarkup() {
  const users = data.users;
  const selectedId = $("#accountSelect")?.value;
  const selected = users.find((user) => user.id === selectedId) || users[0];
  const actions = `${can("accounts", "create") ? btn("Créer un compte", "new-account", "button button-primary", "+") : ""}`;
  const roleDescriptions = { proviseur: "Super-administrateur · accès complet", admin: "Accès étendu · historique protégé", censeur: "Enseignants · notes · bulletins · coefficients", directeur_primaire: "Direction · Crèche & Primaire", directeur_secondaire: "Direction · Secondaire & Lycée", surveillant: "Classes · élèves · présences", econome: "Inscriptions · paiements · dépenses", comptable: "Gestion financière", enseignant: "Classes et notes affectées", accueil: "Inscriptions · lecture élèves", user: "Droits personnalisés" };
  const userRows = users.map((user) => `<tr><td><div class="person-cell"><span class="person-initials">${esc(initials(user.username))}</span><span><strong>${esc(user.username)}</strong><small>${roleName(user.role)}</small></span></div></td><td><span class="status ${user.active ? "" : "inactive"}">${user.active ? "Actif" : "Désactivé"}</span></td><td>${roleDescriptions[user.role] || "Droits personnalisés"}${user.cycle_scope?.length ? `<small class="muted"><br>${user.cycle_scope.join(" · ")}</small>` : ""}</td><td><div class="table-actions">${can("accounts", "update") && (user.role !== "proviseur" || currentUser()?.role === "proviseur") ? `<button class="button button-quiet" data-action="edit-account" data-id="${user.id}">Modifier</button>` : ""}${user.permissions?.accounts?.activate && user.id !== currentUserId && (user.role !== "proviseur" || currentUser()?.role === "proviseur") ? `<button class="button button-quiet" data-action="toggle-account" data-id="${user.id}">${user.active ? "Désactiver" : "Activer"}</button>` : ""}${can("accounts", "delete") && user.id !== currentUserId && user.role !== "proviseur" ? `<button class="icon-button" data-action="delete-account" data-id="${user.id}" aria-label="Supprimer le compte">×</button>` : ""}</div></td></tr>`).join("");
  return `${pageHeading("Comptes & accès", "Gérez les comptes et attribuez les autorisations par rubrique.", actions)}<section class="surface"><div class="surface-header"><div><h2>Comptes utilisateurs</h2><p>Le Proviseur et l’Administrateur peuvent activer ou désactiver les comptes.</p></div></div><div class="table-wrap"><table><thead><tr><th>Utilisateur</th><th>Statut</th><th>Niveau d’accès</th><th>Actions</th></tr></thead><tbody>${userRows}</tbody></table></div></section>${selected && can("accounts", "update") ? `<div class="surface" style="margin-top:16px"><div class="surface-header"><div><h2>Autorisations par rubrique</h2><p>Les modifications sont conservées dès que vous changez une case.</p></div><select id="accountSelect" class="select-control" aria-label="Choisir un compte">${users.map((user) => `<option value="${user.id}" ${user.id === selected.id ? "selected" : ""}>${esc(user.username)}</option>`).join("")}</select></div>${permissionsMarkup(selected)}</div>` : ""}<p class="small-print muted">Les autorisations sont également vérifiées par l’API du serveur.</p>`;
}
function auditMarkup() {
  return `${pageHeading("Journal d’audit", "Historique append-only des connexions et actions sensibles.")}<section class="surface"><div class="surface-header"><div><h2>Événements récents</h2><p>Les entrées ne peuvent pas être supprimées depuis l’application.</p></div></div><div class="table-wrap"><table><thead><tr><th>Date</th><th>Utilisateur</th><th>Action</th><th>Module</th><th>Détail</th></tr></thead><tbody id="auditRows"><tr><td colspan="5">Chargement du journal…</td></tr></tbody></table></div></section>`;
}
async function loadAudit() {
  if (!apiMode) {
    $("#auditRows").innerHTML = `<tr><td colspan="5">Le journal d’audit est disponible avec le serveur Python.</td></tr>`;
    return;
  }
  try {
    const result = await apiRequest("/api/audit?limit=200", { cache: "no-store" });
    const rows = result.entries.map((entry) => `<tr><td>${esc(new Date(entry.created_at).toLocaleString("fr-FR"))}</td><td>${esc(entry.username)}</td><td>${esc(entry.action)}</td><td>${esc(MODULES[entry.module] || entry.module)}</td><td>${esc(entry.summary)}</td></tr>`).join("");
    $("#auditRows").innerHTML = rows || `<tr><td colspan="5">Aucun événement enregistré.</td></tr>`;
  } catch (error) {
    $("#auditRows").innerHTML = `<tr><td colspan="5">${esc(error.message || "Impossible de charger le journal.")}</td></tr>`;
  }
}
function settingsMarkup() {
  const school = data.school;
  const canUpdate = can("settings", "update");
  const isAdministrator = ["admin", "proviseur"].includes(currentUser()?.role);
  return `${pageHeading("Paramètres", "Personnalisez l’application, l’identité de l’établissement et les accès.", `${canUpdate ? btn("Enregistrer les changements", "save-settings", "button button-primary", "✓") : ""}`)}
    <div class="settings-grid"><section class="settings-section"><h2>Application & établissement</h2><p>Ces informations apparaissent dans la navigation et sur les documents scolaires.</p><div class="settings-fields">
      <label class="field full"><span class="field-label">Nom de l’application</span><input class="field-control" name="app-name" value="${esc(school.appName || "Gestion scolaire")}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field full"><span class="field-label">Nom complet de l’établissement</span><input class="field-control" name="school-name" value="${esc(school.name)}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field"><span class="field-label">Nom court</span><input class="field-control" name="school-short" value="${esc(school.shortName)}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field"><span class="field-label">Année scolaire active</span><input class="field-control" name="school-year" value="${esc(school.year)}" disabled><small class="muted small-print">L’année suivante s’active à la clôture du bilan annuel.</small></label>
      <label class="field"><span class="field-label">Seuil de passage (/20)</span><input class="field-control" name="promotion-threshold" type="number" min="0" max="20" step="0.25" value="${esc(school.promotionThreshold ?? 10)}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field"><span class="field-label">Capital initial de conduite</span><select class="field-control" name="conduct-maximum" ${canUpdate ? "" : "disabled"}><option value="18" ${conductCapital() === 18 ? "selected" : ""}>18 points</option><option value="20" ${conductCapital() === 20 ? "selected" : ""}>20 points</option></select><small class="muted small-print">Valeur initiale actuelle de 18 points. Les retraits restent saisis et validés individuellement.</small></label>
      <label class="field full"><span class="field-label">Adresse</span><input class="field-control" name="school-address" value="${esc(school.address)}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field"><span class="field-label">Téléphone</span><input class="field-control" name="school-phone" value="${esc(school.phone)}" ${canUpdate ? "" : "disabled"}></label>
      <label class="field"><span class="field-label">E-mail</span><input class="field-control" type="email" name="school-email" value="${esc(school.email)}" ${canUpdate ? "" : "disabled"}></label>
    </div></section><section class="settings-section"><h2>Logo & palette de couleurs</h2><p>Le logo et la palette sont prévisualisés immédiatement et réutilisés sur les cartes et bulletins.</p><div class="field"><span class="field-label">Logo de l’établissement</span><div class="color-setting"><div class="logo-preview" id="logoPreview">${school.logo ? `<img src="${esc(school.logo)}" alt="Logo actuel">` : esc(initials(school.shortName || school.name))}</div><input type="file" accept=".png,.jpg,.jpeg,.svg,image/png,image/jpeg,image/svg+xml" id="schoolLogoInput" ${canUpdate ? "" : "disabled"} aria-label="Importer le logo"><button class="button button-quiet" data-action="remove-logo" type="button" ${!canUpdate ? "disabled" : ""}>Retirer</button></div></div>
      ${[["schoolColorInput", "Couleur principale", school.color || "#183a4d"], ["schoolSecondaryInput", "Couleur secondaire", school.secondaryColor || "#dce8e4"], ["schoolAccentInput", "Couleur d’accent", school.accentColor || "#c57b53"]].map(([id, label, color]) => `<label class="field"><span class="field-label">${label}</span><div class="color-setting"><input type="color" id="${id}" value="${esc(color)}" ${canUpdate ? "" : "disabled"}><span class="color-swatch-preview" style="background:${esc(color)}"></span></div></label>`).join("")}
      <div class="field"><span class="field-label">Aperçu</span><div class="brand-preview" id="brandPreview"><strong>${esc(school.appName || "Gestion scolaire")}</strong><span>${esc(school.name)}</span></div></div>
      <div class="field"><span class="field-label">Stockage</span><span class="small-print muted">${apiMode ? "Base locale SQLite · accès vérifiés par le serveur" : "Données locales à ce navigateur · exportez une sauvegarde avant de vider les données."}</span></div></section>
    <section class="settings-section"><h2>Compte administrateur</h2><p>Modifiez le nom d’utilisateur ou réinitialisez le mot de passe. Le mot de passe n’est jamais affiché.</p><div class="field"><label class="field-label" for="adminUsername">Nom d’utilisateur</label><input id="adminUsername" class="field-control" value="${esc(currentUser()?.username || "")}" ${isAdministrator ? "" : "disabled"}></div><div class="field"><label class="field-label" for="currentPassword">Mot de passe actuel</label><input id="currentPassword" class="field-control" type="password" autocomplete="current-password" ${isAdministrator ? "" : "disabled"}></div><div class="field"><label class="field-label" for="newPassword">Nouveau mot de passe</label><input id="newPassword" class="field-control" type="password" autocomplete="new-password" placeholder="Laisser vide pour ne pas changer" ${isAdministrator ? "" : "disabled"}></div><button class="button button-secondary" data-action="save-credentials" ${isAdministrator ? "" : "disabled"}>Mettre à jour les identifiants</button></section>
    <section class="settings-section"><h2>Progression & communication</h2><p>Le référentiel couvre Crèche (Petit, Grand), Primaire (1re–6e), Secondaire (7e–9e), puis le Lycée (10e CM, 11e LL/SES/S, 12e TLL/TAL/TSS/TSE/TSEExp/TSEco).</p><label class="field"><span class="field-label">Mode WhatsApp</span><select class="field-control" name="whatsapp-mode" ${canUpdate ? "" : "disabled"}><option value="test" ${school.whatsappMode !== "link" ? "selected" : ""}>Test · aperçu uniquement, aucun envoi</option><option value="link" ${school.whatsappMode === "link" ? "selected" : ""}>Lien WhatsApp · ouverture manuelle</option></select><small class="muted small-print">L’application ne détient aucun secret WhatsApp et ne peut pas envoyer en arrière-plan. Le professeur et la liste sont vérifiés et chaque préparation est auditée.</small></label><label class="field"><span class="field-label">Indicatif téléphonique local</span><input class="field-control" name="country-calling-code" inputmode="numeric" pattern="[0-9]{1,4}" value="${esc(school.countryCallingCode || "223")}" ${canUpdate ? "" : "disabled"}><small class="muted small-print">Les numéros locaux à 8 chiffres reçoivent cet indicatif (Mali : 223).</small></label><p class="small-print muted">Les niveaux et filières sont disponibles dans la fiche de classe. Les voies de Terminale sont des fins de cycle.</p></section>
    ${curriculumEditorMarkup()}
    <section class="settings-section"><h2>Modèle de carte scolaire</h2><p>Choisissez la mise en page et les informations facultatives. Le logo de l’établissement et le drapeau du Mali restent inclus automatiquement.</p><label class="field"><span class="field-label">Disposition</span><select class="field-control" name="card-layout" data-card-template="layout" ${canUpdate ? "" : "disabled"}><option value="horizontal" ${school.cardTemplate.layout !== "vertical" ? "selected" : ""}>Horizontale</option><option value="vertical" ${school.cardTemplate.layout === "vertical" ? "selected" : ""}>Verticale</option></select></label><div class="settings-fields">${[
      ["showPhoto", "Photo de l’élève"], ["showMatricule", "Matricule"], ["showBirthDate", "Date de naissance"], ["showGuardian", "Nom du responsable"], ["showFamilyPhone", "Téléphone familial"],
    ].map(([key, label]) => `<label class="field checkbox-field"><input type="checkbox" name="card-${key}" data-card-template="${key}" ${school.cardTemplate[key] ? "checked" : ""} ${canUpdate ? "" : "disabled"}><span>${label}</span></label>`).join("")}</div><div class="field"><span class="field-label">Aperçu du modèle</span><div id="cardTemplatePreview" class="card-template-preview ${school.cardTemplate.layout === "vertical" ? "vertical" : ""}"><div class="document-identity">${school.logo ? `<img class="document-logo" src="${esc(school.logo)}" alt="">` : `<span class="document-logo-fallback">${esc(initials(school.shortName || school.name))}</span>`}${MALI_FLAG}</div><strong>${esc(school.name)}</strong><span data-preview-field="class">Classe · 5e</span><span data-preview-field="photo">Photo élève</span><span data-preview-field="matricule">Matricule</span><span data-preview-field="birthDate">Date de naissance</span><span data-preview-field="guardian">Responsable</span><span data-preview-field="familyPhone">Téléphone familial</span></div></div></section>
    <section class="settings-section"><h2>Sauvegardes automatiques locales</h2><p>Le serveur Python crée une sauvegarde au démarrage puis toutes les 24 heures. Les 14 dernières sauvegardes sont conservées sur le PC, sans comptes ni mots de passe.</p><p id="automaticBackupStatus" class="small-print muted" role="status">${apiMode ? "Chargement de l’état des sauvegardes…" : "Disponible uniquement avec le serveur Python."}</p><div class="toolbar-group">${apiMode ? `<button class="button button-secondary" data-action="create-automatic-backup" ${canUpdate ? "" : "disabled"}>Créer une sauvegarde maintenant</button>` : ""}</div><p class="small-print muted" style="margin-top:14px">Dossier configurable avec SCHOOL_MANAGEMENT_BACKUP_DIR; fréquence et rétention configurables par variables d’environnement. Le dossier local contient des données personnelles : protégez l’accès au PC.</p></section>
    <section class="settings-section"><h2>Gestion des données</h2><p>Exportez une sauvegarde locale ou restaurez un fichier validé. Les comptes et leurs mots de passe ne sont jamais inclus.</p><div class="toolbar-group"><button class="button button-secondary" data-action="export-data">Télécharger une sauvegarde JSON</button><button class="button button-quiet" data-action="restore-data">Restaurer une sauvegarde</button><input id="backupFileInput" type="file" accept="application/json,.json" hidden><button class="button button-danger" data-action="reset-data" ${currentUser()?.role === "proviseur" ? "" : "disabled"}>Réinitialiser la démonstration</button></div><p class="small-print muted" style="margin-top:14px">Fichier conservé sur le PC de l’établissement. Aucune synchronisation cloud. La restauration conserve les comptes et le journal d’audit.</p></section></div>`;
}
function renderPage({ preserveFocus = false } = {}) {
  if (!currentUser()) return;
  const pageModule = moduleForPage(activePage);
  if (!canRead(pageModule)) {
    activePage = "dashboard";
    if (!canRead("dashboard")) {
      $("#pageContent").innerHTML = `${pageHeading("Accès restreint", "Votre compte n’a pas l’autorisation d’ouvrir les rubriques disponibles.")}<p class="surface empty-state">Demandez à l’administrateur de vérifier vos accès.</p>`;
      return;
    }
  }
  const content = {
    dashboard: dashboardMarkup,
    eleves: studentsMarkup,
    classes: classesMarkup,
    presences: attendanceMarkup,
    matieres: curriculumMarkup,
    dossiers: dossierMarkup,
    enseignants: teachersMarkup,
    discipline: disciplineMarkup,
    paiements: paymentsMarkup,
    depenses: expensesMarkup,
    comptabilite: accountingMarkup,
    bulletins: gradesMarkup,
    cartes: cardsMarkup,
    comptes: accountsMarkup,
    audit: auditMarkup,
    reglages: settingsMarkup,
  }[activePage]();
  $("#pageContent").innerHTML = content;
  const titles = { dashboard: "Vue d’ensemble", eleves: "Élèves", classes: "Classes", presences: "Présences", matieres: "Matières & coefficients", dossiers: "Dossiers élèves", enseignants: "Enseignants & contrats", discipline: "Conduite & incidents", paiements: "Paiements", depenses: "Dépenses", comptabilite: "Comptabilité & rapports", bulletins: "Notes & bulletins", cartes: "Cartes scolaires", comptes: "Comptes & accès", audit: "Journal d’audit", reglages: "Paramètres" };
  $("#breadcrumbPage").textContent = titles[activePage];
  $$(".nav-item").forEach((item) => {
    item.hidden = !canRead(moduleForPage(item.dataset.page));
    item.classList.toggle("active", item.dataset.page === activePage);
    item.setAttribute("aria-current", item.dataset.page === activePage ? "page" : "false");
  });
  $("#studentCountNav").textContent = String(data.students.filter((student) => student.active).length);
  $("#breadcrumbSchool").textContent = data.school.shortName || data.school.name;
  $("#sidebarSchoolName").textContent = data.school.shortName || data.school.name;
  $("#loginSchoolName").textContent = data.school.appName || data.school.name;
  $("#currentUserName").textContent = currentUser().username;
  $("#currentUserRole").textContent = roleName(currentUser().role);
  $("#userAvatar").textContent = initials(currentUser().username);
  $(".school-year").textContent = data.school.year;
  document.title = `${data.school.appName || "Gestion scolaire"} · ${data.school.name}`;
  $(".brand-copy span").textContent = (data.school.appName || "Gestion scolaire").toLocaleUpperCase("fr");
  document.documentElement.style.setProperty("--school-color", data.school.color || "#183a4d");
  document.documentElement.style.setProperty("--school-color-secondary", data.school.secondaryColor || "#dce8e4");
  document.documentElement.style.setProperty("--school-color-accent", data.school.accentColor || "#c57b53");
  if (activePage === "audit") loadAudit();
  if (activePage === "reglages" && apiMode && can("settings", "read")) refreshAutomationStatus();
  if (preserveFocus) $("#pageContent").focus({ preventScroll: true });
}
function navigate(page) {
  if (!moduleForPage(page) || !canRead(moduleForPage(page))) {
    showToast("Votre compte n’a pas accès à cette rubrique.");
    return;
  }
  activePage = page;
  $("#sidebar").classList.remove("open");
  $("#sidebarScrim").classList.remove("active");
  renderPage();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function requirePermission(module, action) {
  if (!can(module, action)) {
    showToast("Cette action n’est pas autorisée pour votre compte.");
    return false;
  }
  return true;
}
function field(label, name, value = "", type = "text", options = {}) {
  const { required = true, full = false, choices, min, max, step, placeholder } = options;
  const input = choices
    ? `<select class="field-control" name="${name}" ${required ? "required" : ""}>${choices.map((choice) => `<option value="${esc(choice.value)}" ${String(choice.value) === String(value) ? "selected" : ""}>${esc(choice.label)}</option>`).join("")}</select>`
    : type === "textarea"
      ? `<textarea class="field-control" name="${name}" ${required ? "required" : ""} placeholder="${esc(placeholder || "")}">${esc(value)}</textarea>`
      : `<input class="field-control" name="${name}" type="${type}" value="${esc(value)}" ${required ? "required" : ""} ${min !== undefined ? `min="${min}"` : ""} ${max !== undefined ? `max="${max}"` : ""} ${step ? `step="${step}"` : ""} placeholder="${esc(placeholder || "")}" ${type === "password" ? 'autocomplete="new-password"' : ""}>`;
  return `<label class="field ${full ? "full" : ""}"><span class="field-label">${label}</span>${input}</label>`;
}
function valuesForDirectorRole(role) {
  if (role === "directeur_primaire") return ["Crèche", "Primaire"];
  if (role === "directeur_secondaire") return ["Secondaire", "Lycée"];
  return null;
}
function syncCycleScopeOptions(role, preferred = []) {
  const select = $('#dialogFields [name="cycle_scope"]');
  if (!select) return;
  const directorCycles = valuesForDirectorRole(role);
  const cycles = directorCycles || ["Crèche", "Primaire", "Secondaire", "Lycée"];
  const selected = directorCycles || preferred;
  select.innerHTML = cycles.map((cycle) => `<option value="${cycle}" ${selected.includes(cycle) ? "selected" : ""}>${cycle}</option>`).join("");
}
function studentClassChoices(selected) {
  const classes = data.classes.filter((item) => item.active).map((item) => ({ value: item.name, label: item.name }));
  if (selected && !classes.some((item) => item.value === selected)) classes.push({ value: selected, label: selected });
  return classes;
}
function subjectChoicesForClass(className, selected = "") {
  const configured = subjectsForLevel(academicLevel(data.classes.find((item) => item.name === className) || {})).map((item) => item.name);
  const names = configured.length ? configured : ["Français", "Mathématiques", "Sciences", "Histoire-Géo", "Anglais", "Éducation physique", "Autre"];
  if (selected && !names.includes(selected)) names.push(selected);
  return names.map((name) => ({ value: name, label: name }));
}
function teacherChoices(selected) {
  const teachers = data.teachers.map((teacher) => ({ value: teacher.id, label: `${teacher.firstName} ${teacher.lastName}` }));
  if (selected && !teachers.some((teacher) => teacher.value === selected)) teachers.push({ value: selected, label: selected });
  return teachers;
}
function academicLevelChoices(selected = "") {
  const choices = Object.entries(data.school.academicCycles || ACADEMIC_CYCLES).flatMap(([cycle, levels]) => levels.map((level) => ({ value: level, label: `${cycle} · ${level}` })));
  if (selected && !choices.some((item) => item.value === selected)) choices.unshift({ value: selected, label: `${academicLabel(selected)} · niveau historique` });
  return choices;
}
function classesAtNextLevels(level) {
  const levels = NEXT_LEVELS[level] || [];
  return data.classes.filter((item) => item.active && levels.includes(academicLevel(item)));
}
function annualDecisionFor(student, year) {
  const average = annualAverageFor(student.id, year);
  const level = academicLevel(data.classes.find((item) => item.name === enrollmentForYear(student, year)?.className) || {});
  if (average === null) return "missing";
  if (average < Number(data.school.promotionThreshold ?? 10)) return "repeat";
  return TERMINAL_LEVELS.has(level) ? "graduate" : "promote";
}
function openRecord(kind, id = null) {
  const existing = id ? ({
    student: studentById(id),
    class: data.classes.find((item) => item.id === id),
    payment: data.payments.find((item) => item.id === id),
    expense: data.expenses.find((item) => item.id === id),
    grade: data.grades.find((item) => item.id === id),
    teacher: data.teachers.find((item) => item.id === id),
    assessment: data.assessments.find((item) => item.id === id),
    incident: data.incidents.find((item) => item.id === id),
    account: data.users.find((item) => item.id === id),
  }[kind]) : null;
  const gradeStudentId = existing?.studentId || $("#gradeStudent")?.value || data.students.find((student) => student.active)?.id || "";
  const gradeStudentClass = data.students.find((student) => student.id === gradeStudentId)?.className;
  const gradeSubjects = subjectChoicesForClass(gradeStudentClass, existing?.subject || "");
  const defaultGradeSubject = existing?.subject || gradeSubjects[0]?.value || "";
  const assessmentClassName = existing?.className || data.classes.find((item) => item.active)?.name || "";
  const specs = {
    student: {
      module: "students", title: existing ? "Modifier le dossier" : "Nouvelle inscription",
      fields: [
        field("Prénom", "firstName", existing?.firstName || ""),
        field("Nom de famille", "lastName", existing?.lastName || ""),
        field("Genre", "gender", existing?.gender || "F", "text", { choices: [{ value: "F", label: "Féminin" }, { value: "M", label: "Masculin" }] }),
        field("Date de naissance", "birthDate", existing?.birthDate || "", "date"),
        field("Classe", "className", existing?.className || "", "text", { choices: studentClassChoices(existing?.className) }),
        field("Date d’inscription", "enrollmentDate", existing?.enrollmentDate || today(), "date"),
        field("Frais scolaires annuels (FCFA)", "annualFee", existing?.annualFee ?? "", "number", { min: 0, step: 1 }),
        field("Nom du parent / responsable", "guardian", existing?.guardian || ""),
        field("Téléphone du responsable", "phone", existing?.phone || "", "tel", { required: false }),
      ],
      extraFields: `<label class="field full"><span class="field-label">Photo de l’élève pour sa carte scolaire</span><input id="studentPhotoInput" class="field-control" type="file" accept="image/*"><small class="muted small-print">Image JPEG, PNG ou WebP · 1 Mo maximum</small><span id="studentPhotoPreview" class="photo-preview">${existing?.photo ? `<img src="${existing.photo}" alt="Photo actuelle">` : "Aucune photo sélectionnée"}</span></label>`,
    },
    class: {
      module: "classes", title: existing ? "Modifier la classe" : "Créer une classe",
      fields: [
        field("Nom de la classe", "name", existing?.name || ""),
        field("Cycle", "level", existing?.level || "Primaire", "text", { choices: Object.keys(data.school.academicCycles || ACADEMIC_CYCLES).map((cycle) => ({ value: cycle, label: cycle })) }),
        field("Niveau / filière", "academicLevel", academicLevel(existing || {}), "text", { choices: academicLevelChoices(academicLevel(existing || {})) }),
        field("Effectif maximum", "capacity", existing?.capacity ?? 30, "number", { min: 1 }),
        field("Frais annuels (FCFA)", "annualFee", existing?.annualFee ?? "", "number", { min: 0, step: 1 }),
        field("Professeur titulaire", "teacherId", existing?.teacherId || "", "text", { required: false, choices: [{ value: "", label: "Affectation à préciser" }, ...teacherChoices(existing?.teacherId)] }),
        field("Statut", "active", existing?.active === false ? "false" : "true", "text", { choices: [{ value: "true", label: "Active" }, { value: "false", label: "Inactive" }] }),
      ],
    },
    payment: {
      module: "payments", title: existing ? "Modifier le paiement" : "Enregistrer un paiement",
      fields: [
        field("Élève", "studentId", existing?.studentId || data.students.find((student) => student.active)?.id || "", "text", { choices: data.students.filter((student) => student.active).map((student) => ({ value: student.id, label: `${student.firstName} ${student.lastName} · ${student.className}` })) }),
        field("Montant (FCFA)", "amount", existing?.amount ?? "", "number", { min: 1, step: 1 }),
        field("Moyen de paiement", "method", existing?.method || METHODS[0], "text", { choices: METHODS.map((method) => ({ value: method, label: method })) }),
        field("Date", "date", existing?.date || today(), "date"),
        field("Référence du reçu", "reference", existing?.reference || `REC-${String(data.payments.length + 105).padStart(3, "0")}`, "text", { required: false }),
        field("Note", "note", existing?.note || "", "text", { required: false }),
      ],
    },
    expense: {
      module: "expenses", title: existing ? "Modifier la dépense" : "Enregistrer une dépense",
      fields: [
        field("Libellé", "label", existing?.label || ""),
        field("Catégorie", "category", existing?.category || "Fournitures", "text", { choices: [...new Set([existing?.category, "Cahiers", "Stylos / bics", "Craies", "Tenues scolaires", "Fournitures", "Personnel", "Entretien", "Charges", "Transport", "Autre"].filter(Boolean))].map((item) => ({ value: item, label: item })) }),
        field("Montant (FCFA)", "amount", existing?.amount ?? "", "number", { min: 1, step: 1 }),
        field("Date", "date", existing?.date || today(), "date"),
        field("Description / justificatif", "note", existing?.note || "", "textarea", { required: false, full: true }),
      ],
    },
    grade: {
      module: "grades", title: "Ajouter une note",
      fields: [
        field("Élève", "studentId", gradeStudentId, "text", { choices: data.students.filter((student) => student.active).map((student) => ({ value: student.id, label: `${student.firstName} ${student.lastName} · ${student.className}` })) }),
        field("Matière", "subject", defaultGradeSubject, "text", { choices: gradeSubjects }),
        field("Note / 20", "score", existing?.score ?? "", "number", { min: 0, max: 20, step: .25 }),
        field("Coefficient", "coefficient", existing?.coefficient ?? coefficientForClassSubject(gradeStudentClass, defaultGradeSubject), "number", { min: .25, step: .25 }),
        field("Trimestre", "term", existing?.term || "1er trimestre", "text", { choices: ["1er trimestre", "2e trimestre", "3e trimestre"].map((item) => ({ value: item, label: item })) }),
      ],
    },
    teacher: {
      module: "teachers", title: existing ? "Modifier l’enseignant" : "Nouvel enseignant · contrat",
      fields: [
        field("Prénom", "firstName", existing?.firstName || ""),
        field("Nom", "lastName", existing?.lastName || ""),
        field("Matière / spécialité", "subject", existing?.subject || ""),
        field("Téléphone", "phone", existing?.phone || "", "tel", { required: false }),
        field("Type de contrat", "contractType", existing?.contractType || "CDD", "text", { choices: ["CDI", "CDD", "Vacataire", "Stage"].map((item) => ({ value: item, label: item })) }),
        field("Date de début", "startDate", existing?.startDate || today(), "date"),
        field("Date d’échéance", "endDate", existing?.endDate || "", "date", { required: false }),
        field("Salaire mensuel (FCFA)", "salary", existing?.salary ?? "", "number", { min: 0, step: 1 }),
        field("Statut", "status", existing?.status || "Actif", "text", { choices: [{ value: "Actif", label: "Actif" }, { value: "Suspendu", label: "Suspendu" }, { value: "Parti", label: "Parti" }] }),
      ],
      extraFields: `<label class="field full"><span class="field-label">Classes affectées</span><select class="field-control" name="classIds" multiple size="5">${data.classes.filter((item) => item.active).map((item) => `<option value="${esc(item.id)}" ${item.teacherId === existing?.id ? "selected" : ""}>${esc(item.name)} · ${esc(academicLevel(item) || item.level)}</option>`).join("")}</select><small class="muted small-print">Le professeur titulaire et sa liste annuelle sont associés aux classes sélectionnées.</small></label>`,
    },
    assessment: {
      module: "grades", title: "Créer une évaluation de classe",
      fields: [
        field("Classe", "className", assessmentClassName, "text", { choices: data.classes.filter((item) => item.active).map((item) => ({ value: item.name, label: item.name })) }),
        field("Matière", "subject", subjectChoicesForClass(assessmentClassName)[0]?.value || "", "text", { choices: subjectChoicesForClass(assessmentClassName) }),
        field("Type", "type", "Composition", "text", { choices: ["Devoir", "Composition", "Examen"].map((item) => ({ value: item, label: item })) }),
        field("Intitulé", "title", "", "text", { required: false, placeholder: "Ex. Composition du 1er trimestre" }),
        field("Date", "date", today(), "date"),
        field("Trimestre", "term", selectedGradeTerm, "text", { choices: ["1er trimestre", "2e trimestre", "3e trimestre"].map((item) => ({ value: item, label: item })) }),
      ],
    },
    incident: {
      module: "discipline", title: "Signaler un incident",
      fields: [
        field("Élève concerné", "studentId", id || data.students.find((student) => student.active)?.id || "", "text", { choices: data.students.filter((student) => student.active).map((student) => ({ value: student.id, label: `${student.firstName} ${student.lastName} · ${student.className}` })) }),
        field("Date de l’incident", "date", today(), "date"),
        field("Motif", "reason", "", "textarea", { full: true }),
        field("Points à retirer", "pointsRemoved", "", "number", { min: 0.01, step: 0.25, placeholder: "Saisie manuelle · aucun barème automatique" }),
        field("Observation", "observation", "", "textarea", { required: false, full: true }),
      ],
    },
    account: {
      module: "accounts", title: existing ? "Modifier le compte" : "Créer un compte",
      fields: [
        field("Nom d’utilisateur", "username", existing?.username || ""),
        field(existing ? "Nouveau mot de passe (facultatif)" : "Mot de passe", "password", "", "password", { required: !existing, full: true, placeholder: existing ? "Laisser vide pour ne pas changer" : "" }),
        currentUser()?.role === "proviseur"
          ? field("Rôle", "role", existing?.role || "user", "text", { choices: [{ value: "proviseur", label: "Proviseur · super-administrateur" }, { value: "admin", label: "Administrateur" }, { value: "censeur", label: "Censeur" }, { value: "directeur_primaire", label: "Direction · Crèche & Primaire" }, { value: "directeur_secondaire", label: "Direction · Secondaire & Lycée" }, { value: "surveillant", label: "Surveillant" }, { value: "econome", label: "Économe" }, { value: "comptable", label: "Comptable" }, { value: "enseignant", label: "Enseignant" }, { value: "accueil", label: "Agent d’accueil" }, { value: "user", label: "Rôle personnalisé" }] })
          : `<input type="hidden" name="role" value="${esc(existing?.role || "user")}"><p class="small-print muted">Rôle attribué : ${roleName(existing?.role || "user")} · Seul le Proviseur peut attribuer ou modifier les rôles.</p>`,
        `<label class="field full"><span class="field-label">Périmètre des classes</span><select class="field-control" name="class_scope" multiple size="4">${data.classes.map((item) => `<option value="${esc(item.name)}" ${(existing?.class_scope || []).includes(item.name) ? "selected" : ""}>${esc(item.name)}</option>`).join("")}</select><small class="muted small-print">Obligatoire pour le rôle Enseignant. Sans sélection, les autres rôles utilisent leurs permissions générales.</small></label>`,
        `<label class="field full"><span class="field-label">Périmètre des cycles</span><select class="field-control" name="cycle_scope" multiple size="4">${["Crèche", "Primaire", "Secondaire", "Lycée"].map((cycle) => { const fixed = valuesForDirectorRole(existing?.role); const selected = fixed ? fixed.includes(cycle) : (existing?.cycle_scope || []).includes(cycle); return `<option value="${cycle}" ${selected ? "selected" : ""}>${cycle}</option>`; }).join("")}</select><small class="muted small-print">Les comptes de direction sont limités à leur cycle : Crèche/Primaire ou Secondaire/Lycée. Un compte peut aussi être restreint à des classes précises.</small></label>`,
      ],
    },
  };
  const spec = specs[kind];
  if (!spec || !requirePermission(spec.module, existing ? "update" : "create")) return;
  if (kind === "account" && existing?.role === "proviseur" && currentUser()?.role !== "proviseur") {
    showToast("Seul le Proviseur peut modifier ce compte.");
    return;
  }
  dialogState = { kind, id, module: spec.module };
  dialogPhoto = existing?.photo || "";
  $("#dialogTitle").textContent = spec.title;
  $("#dialogKicker").textContent = existing ? "MODIFICATION" : "NOUVEAU DOSSIER";
  $("#dialogFields").innerHTML = [...spec.fields, spec.extraFields || ""].join("");
  if (kind === "account") {
    const roleInput = $('#dialogFields [name="role"]');
    const scopeInput = $('#dialogFields [name="class_scope"]');
    scopeInput.required = roleInput.value === "enseignant";
    syncCycleScopeOptions(roleInput.value, existing?.cycle_scope || []);
  }
  $("#dialogError").hidden = true;
  $("#recordDialog").showModal();
}
async function saveRecord(values) {
  const { kind, id } = dialogState;
  const existing = id;
  const collectionName = ({ student: "students", class: "classes", payment: "payments", expense: "expenses", grade: "grades", assessment: "assessments", incident: "incidents", teacher: "teachers", account: "users" })[kind];
  const list = data[collectionName];
  let record = existing ? list.find((item) => item.id === id) : { id: uid() };
  if (!record) throw new Error("Le dossier à modifier n’existe plus.");
  if (kind === "class" && values.name !== record.name && data.classes.some((item) => item.name.toLowerCase() === values.name.toLowerCase())) throw new Error("Une classe portant ce nom existe déjà.");
  if (kind === "student") {
    values.annualFee = Number(values.annualFee);
    values.active = record.active ?? true;
    values.photo = dialogPhoto;
    values.matricule = record.matricule || nextMatricule();
    values.enrollmentYear = record.enrollmentYear || data.school.year;
    values.enrollmentHistory = [...(record.enrollmentHistory || [])];
    const currentEnrollment = values.enrollmentHistory.find((item) => item.schoolYear === data.school.year);
    if (currentEnrollment) currentEnrollment.className = values.className;
    else values.enrollmentHistory.push({ schoolYear: data.school.year, className: values.className, decision: existing ? "transfert" : "inscrit" });
  }
  if (kind === "class") {
    values.capacity = Number(values.capacity);
    values.annualFee = Number(values.annualFee);
    values.active = values.active === "true";
    values.teacherId = values.teacherId || "";
    const teacher = data.teachers.find((item) => item.id === values.teacherId);
    values.teacher = teacher ? `${teacher.firstName} ${teacher.lastName}` : "";
    if (existing && record.name !== values.name) {
      data.students.forEach((student) => {
        if (student.className === record.name) student.className = values.name;
        const currentEnrollment = (student.enrollmentHistory || []).find((item) => item.schoolYear === data.school.year);
        if (currentEnrollment?.className === record.name) currentEnrollment.className = values.name;
      });
    }
  }
  if (kind === "payment" || kind === "expense") values.amount = Number(values.amount);
  if (kind === "grade") {
    values.score = Number(values.score);
    values.coefficient = Number(values.coefficient);
    values.term = selectedGradeTerm || values.term;
    values.schoolYear = record.schoolYear || data.school.year;
    if (values.score < 0 || values.score > 20) throw new Error("La note doit être comprise entre 0 et 20.");
    if (!Number.isFinite(values.coefficient) || values.coefficient <= 0) throw new Error("Le coefficient doit être un nombre positif.");
  }
  if (kind === "assessment") {
    const students = data.students.filter((student) => student.active && student.className === values.className);
    if (!students.length) throw new Error("Cette classe ne contient aucun élève actif.");
    values.title = values.title.trim() || `${values.type} · ${values.subject}`;
    values.createdAt = today();
    values.schoolYear = data.school.year;
  }
  if (kind === "incident") {
    values.pointsRemoved = Number(values.pointsRemoved);
    if (!Number.isFinite(values.pointsRemoved) || values.pointsRemoved <= 0) throw new Error("Saisissez un retrait positif; aucun barème automatique n’est appliqué.");
    values.validated = false;
    values.author = currentUser()?.username || "Personnel habilité";
    values.validatedBy = "";
    values.validatedAt = "";
  }
  if (kind === "teacher") {
    values.salary = Number(values.salary || 0);
    values.classIds = values.classIds || [];
    if (existing) {
      const previousName = `${record.firstName} ${record.lastName}`;
      const nextName = `${values.firstName} ${values.lastName}`;
      data.classes.forEach((schoolClass) => {
        if (schoolClass.teacherId === record.id || schoolClass.teacher === previousName) {
          schoolClass.teacher = nextName;
        }
      });
    }
  }
  if (kind === "account") {
    values.class_scope = values.class_scope || [];
    values.cycle_scope = values.cycle_scope || [];
    if (values.role === "enseignant" && values.class_scope.length === 0) throw new Error("Attribuez au moins une classe à l’enseignant.");
    const usernameExists = data.users.some((user) => user.id !== id && user.username.toLowerCase() === values.username.toLowerCase());
    if (usernameExists) throw new Error("Ce nom d’utilisateur est déjà utilisé.");
    if (record.role === "proviseur" && currentUser()?.role !== "proviseur") throw new Error("Seul le Proviseur peut modifier ce compte.");
    const password = values.password;
    if (password && password.length < 8) throw new Error("Le mot de passe doit contenir au moins 8 caractères.");
    if (existing && record.id === currentUserId && values.role !== record.role) throw new Error("Vous ne pouvez pas modifier votre propre rôle.");
    if (apiMode) {
      const payload = { username: values.username, class_scope: values.class_scope, cycle_scope: values.cycle_scope };
      if (!existing || currentUser()?.role === "proviseur") payload.role = values.role;
      if (password) payload.password = password;
      const savedUser = await apiRequest(existing ? `/api/accounts/${encodeURIComponent(id)}` : "/api/accounts", {
        method: existing ? "PATCH" : "POST",
        body: JSON.stringify(payload),
      });
      if (existing) data.users = data.users.map((item) => item.id === id ? savedUser : item);
      else data.users.push(savedUser);
      $("#recordDialog").close();
      renderPage();
      showToast(existing ? "Compte mis à jour." : "Compte créé avec les droits du rôle sélectionné.");
      return;
    }
    delete values.password;
    if (existing && password) Object.assign(record, await passwordDigest(password));
    if (!existing) {
      Object.assign(record, await passwordDigest(password));
      record.active = true;
      record.permissions = permissionsForRole(values.role);
    }
    if (existing && values.role !== record.role) record.permissions = permissionsForRole(values.role);
  }
  Object.assign(record, values);
  if (!existing) list.push(record);
  if (kind === "teacher") {
    data.classes.forEach((schoolClass) => {
      if (values.classIds.includes(schoolClass.id)) {
        schoolClass.teacherId = record.id;
        schoolClass.teacher = `${record.firstName} ${record.lastName}`;
      } else if (schoolClass.teacherId === record.id) {
        schoolClass.teacherId = "";
        schoolClass.teacher = "";
      }
    });
    delete record.classIds;
  }
  if (kind === "assessment" && !existing) {
    data.grades.push(...data.students.filter((student) => student.active && student.className === record.className).map((student) => ({
      id: uid(), studentId: student.id, assessmentId: record.id, subject: record.subject,
      score: null, coefficient: coefficientForClassSubject(record.className, record.subject), term: record.term, schoolYear: record.schoolYear, type: `${record.type} · ${record.title}`,
    })));
  }
  persist();
  $("#recordDialog").close();
  renderPage();
  showToast(kind === "student" ? "Fiche d’élève enregistrée." : kind === "payment" ? "Paiement enregistré et crédit recalculé." : kind === "expense" ? "Dépense enregistrée." : kind === "grade" ? "Note enregistrée. La moyenne a été recalculée." : kind === "assessment" ? "Évaluation créée pour les élèves actifs de la classe." : kind === "incident" ? "Incident signalé; les points restent en attente de validation." : kind === "teacher" ? "Fiche enseignant et contrat enregistrés." : "Modifications enregistrées.");
}
async function deleteRecord(kind, id) {
  const permissions = { student: ["students", "élève"], class: ["classes", "classe"], payment: ["payments", "paiement"], expense: ["expenses", "dépense"], grade: ["grades", "note"], assessment: ["grades", "évaluation"], incident: ["discipline", "incident"], teacher: ["teachers", "enseignant"], account: ["accounts", "compte"] }[kind];
  if (!permissions || !requirePermission(permissions[0], "delete")) return;
  const record = (kind === "student" ? data.students : kind === "class" ? data.classes : kind === "payment" ? data.payments : kind === "expense" ? data.expenses : kind === "grade" ? data.grades : kind === "assessment" ? data.assessments : kind === "incident" ? data.incidents : kind === "teacher" ? data.teachers : data.users).find((item) => item.id === id);
  if (!record) return;
  if (kind === "incident" && record.validated) {
    showToast("Un incident validé est conservé dans l’historique.");
    return;
  }
  if (kind === "class" && classSize(record.name) > 0) {
    showToast("Cette classe contient encore des élèves. Réaffectez-les avant de la supprimer.");
    return;
  }
  if (kind === "account" && (record.id === currentUserId || record.role === "proviseur")) {
    showToast("Ce compte ne peut pas être supprimé.");
    return;
  }
  if (kind === "grade" && record.assessmentId) {
    record.score = null;
    persist();
    renderPage();
    showToast("Note effacée. L’élève reste dans la grille de composition.");
    return;
  }
  if (!window.confirm(`Supprimer ce dossier${kind === "student" ? ` (${record.firstName} ${record.lastName})` : ""} ? Cette action est définitive.`)) return;
  if (kind === "account" && apiMode) {
    try {
      await apiRequest(`/api/accounts/${encodeURIComponent(id)}`, { method: "DELETE" });
      data.users = data.users.filter((item) => item.id !== id);
      renderPage();
      showToast("Compte supprimé.");
    } catch (error) {
      showToast(error.message || "Impossible de supprimer ce compte.");
    }
    return;
  }
  const collection = kind === "student" ? data.students : kind === "class" ? data.classes : kind === "payment" ? data.payments : kind === "expense" ? data.expenses : kind === "grade" ? data.grades : kind === "assessment" ? data.assessments : kind === "incident" ? data.incidents : kind === "teacher" ? data.teachers : data.users;
  data[({ student: "students", class: "classes", payment: "payments", expense: "expenses", grade: "grades", assessment: "assessments", incident: "incidents", teacher: "teachers", account: "users" })[kind]] = collection.filter((item) => item.id !== id);
  if (kind === "student") {
    data.payments = data.payments.filter((item) => item.studentId !== id);
    data.grades = data.grades.filter((item) => item.studentId !== id);
  }
  if (kind === "assessment") data.grades = data.grades.filter((grade) => grade.assessmentId !== id);
  if (kind === "teacher") data.classes.forEach((schoolClass) => {
    if (schoolClass.teacherId === id) {
      schoolClass.teacherId = "";
      schoolClass.teacher = "";
    }
  });
  persist();
  renderPage();
  showToast("Dossier supprimé.");
}
function printReport(studentId) {
  if (!requirePermission("grades", "read")) return;
  const student = studentById(studentId);
  if (!student) return;
  const container = $("#pageContent");
  const old = container.innerHTML;
  container.innerHTML = reportMarkup(student);
  window.print();
  container.innerHTML = old;
}
function printEvaluationHistory(studentId) {
  if (!requirePermission("grades", "read")) return;
  const student = studentById(studentId);
  if (!student) return;
  const container = $("#pageContent");
  const old = container.innerHTML;
  container.innerHTML = evaluationHistoryMarkup(student);
  window.print();
  container.innerHTML = old;
}
async function validateIncident(id) {
  if (!requirePermission("discipline", "update")) return;
  const incident = data.incidents.find((item) => item.id === id);
  if (!incident || incident.validated) return;
  if (!window.confirm(`Valider le retrait manuel de ${incident.pointsRemoved} point(s) pour ${studentName(incident.studentId)} ?`)) return;
  if (apiMode) {
    try {
      const result = await apiRequest(`/api/discipline/incidents/${encodeURIComponent(id)}/validate`, { method: "POST" });
      Object.assign(incident, result.incident);
    } catch (error) {
      showToast(error.message || "Impossible de valider cet incident.");
      return;
    }
  } else {
    incident.validated = true;
    incident.validatedBy = currentUser()?.username || "Personnel habilité";
    incident.validatedAt = new Date().toISOString();
    persist();
  }
  renderPage();
  showToast(`Retrait validé. Note de conduite : ${conductBalance(incident.studentId)} / ${conductCapital()} points.`);
}
async function toggleAccount(id) {
  const user = data.users.find((item) => item.id === id);
  if (!user || !can("accounts", "activate") || user.role === "proviseur" && currentUser()?.role !== "proviseur" || user.id === currentUserId) {
    showToast("Votre compte ne peut pas activer cet utilisateur; seul le Proviseur peut gérer un autre compte Proviseur.");
    return;
  }
  if (apiMode) {
    try {
      await apiRequest(`/api/accounts/${encodeURIComponent(id)}/activation`, {
        method: "PATCH",
        body: JSON.stringify({ active: !user.active }),
      });
      user.active = !user.active;
    } catch (error) {
      showToast(error.message || "Impossible de modifier le statut du compte.");
      return;
    }
  } else user.active = !user.active;
  if (!apiMode) persist();
  renderPage();
  showToast(user.active ? "Compte activé." : "Compte désactivé.");
}
async function updatePermission(event) {
  const input = event.target.closest("[data-permission]");
  if (!input) return;
  if (!requirePermission("accounts", "update")) {
    input.checked = !input.checked;
    return;
  }
  const [module, action] = input.dataset.permission.split(".");
  const user = data.users.find((item) => item.id === $("#accountSelect")?.value);
  if (!user || user.role === "proviseur" || user.id === currentUserId) return;
  const previousValue = Boolean(user.permissions?.[module]?.[action]);
  user.permissions[module] ||= {};
  user.permissions[module][action] = input.checked;
  let saved = true;
  if (apiMode) {
    try {
      const result = await apiRequest(`/api/accounts/${encodeURIComponent(user.id)}`, {
        method: "PATCH",
        body: JSON.stringify({ permissions: user.permissions }),
      });
      Object.assign(user, result);
    } catch (error) {
      user.permissions[module][action] = previousValue;
      saved = false;
      showToast(error.message || "Impossible d’enregistrer cette autorisation.");
    }
  } else persist();
  renderPage();
  if (saved) showToast("Autorisation mise à jour.");
}
function handleSearch(event) {
  if (!event.target.matches('[data-search="students"]')) return;
  const query = event.target.value.trim().toLocaleLowerCase("fr");
  const className = $('[data-filter="student-class"]')?.value || "";
  const results = data.students.filter((student) => {
    const text = `${student.firstName} ${student.lastName} ${student.guardian} ${student.className}`.toLocaleLowerCase("fr");
    return (!query || text.includes(query)) && (!className || student.className === className);
  });
  $("#studentRows").innerHTML = studentRows(results);
}
function printClassRoster(classId) {
  if (!requirePermission("students", "read")) return;
  const schoolClass = data.classes.find((item) => item.id === classId);
  if (!schoolClass) return;
  const roster = studentsInClass(schoolClass.name, selectedAcademicYear || data.school.year);
  const container = $("#pageContent");
  const old = container.innerHTML;
  container.innerHTML = `<article class="report-sheet"><div class="report-school"><div class="brand-emblem">${data.school.logo ? `<img src="${esc(data.school.logo)}" alt="">` : esc(initials(data.school.shortName || data.school.name))}</div><div><h2>${esc(data.school.name)}</h2><p>${esc(data.school.address)} · ${esc(data.school.phone)}</p></div></div><h3 class="report-title">LISTE DES ÉLÈVES · ${esc(schoolClass.name)}</h3><p>Année scolaire : ${esc(selectedAcademicYear || data.school.year)} · Professeur : ${esc(data.teachers.find((item) => item.id === schoolClass.teacherId)?.firstName || "")} ${esc(data.teachers.find((item) => item.id === schoolClass.teacherId)?.lastName || "")}</p><table class="report-table"><thead><tr><th>N°</th><th>Matricule</th><th>Nom et prénom</th><th>Responsable</th><th>Contact</th></tr></thead><tbody>${roster.map((student, index) => `<tr><td>${index + 1}</td><td>${esc(student.matricule || "—")}</td><td>${esc(student.lastName)} ${esc(student.firstName)}</td><td>${esc(student.guardian || "—")}</td><td>${esc(student.phone || "—")}</td></tr>`).join("")}</tbody></table><p>${roster.length} élève${roster.length === 1 ? "" : "s"}</p></article>`;
  window.print();
  container.innerHTML = old;
}
async function closeAcademicYear() {
  if (!apiMode) {
    showToast("La clôture avec historique sécurisé requiert le serveur FastAPI.");
    return;
  }
  const year = data.school.year;
  const students = data.students.filter((student) => student.active && enrollmentForYear(student, year));
  const decisions = [];
  for (const student of students) {
    const decision = promotionDraft[student.id]?.decision || annualDecisionFor(student, year);
    if (decision === "missing") throw new Error(`Saisissez toutes les notes annuelles de ${student.firstName} ${student.lastName} avant la clôture.`);
    let targetClassId = null;
    if (decision === "promote") {
      const currentClass = data.classes.find((item) => item.name === enrollmentForYear(student, year)?.className);
      const targets = classesAtNextLevels(academicLevel(currentClass || {}));
      targetClassId = promotionDraft[student.id]?.targetClassId || targets[0]?.id || null;
      if (!targetClassId) throw new Error(`Créez une classe du niveau suivant avant de passer ${student.firstName} ${student.lastName}.`);
    }
    decisions.push({ student_id: student.id, decision, target_class_id: targetClassId });
  }
  if (!students.length) throw new Error("Aucun élève inscrit à clôturer pour cette année.");
  if (!window.confirm(`Clôturer ${year}, enregistrer ${decisions.length} décisions et activer l’année ${_nextYearLabel(year)} ? Cette opération conserve l’historique et ne peut pas être annulée depuis l’application.`)) return;
  const result = await apiRequest("/api/academic/close-year", {
    method: "POST",
    body: JSON.stringify({ school_year: year, decisions }),
  });
  const state = await apiRequest("/api/state", { cache: "no-store" });
  if (!state.state) throw new Error("La clôture est enregistrée, mais l’état scolaire ne peut pas être rechargé.");
  data = { ...state.state, analytics: state.analytics, users: data.users };
  ensureCollections();
  selectedAcademicYear = result.school_year;
  promotionDraft = {};
  renderPage();
  showToast(`Bilan clôturé; listes actualisées pour ${result.school_year}.`);
}
function _nextYearLabel(year) {
  const match = year.match(/^(\d{4})\s*[—–-]\s*(\d{4})$/);
  return match ? `${Number(match[1]) + 1} — ${Number(match[2]) + 1}` : "l’année suivante";
}
async function prepareWhatsAppRoster(classId) {
  if (!apiMode) {
    showToast("La préparation WhatsApp traçable nécessite le serveur FastAPI.");
    return;
  }
  const prepared = await apiRequest("/api/whatsapp/roster", {
    method: "POST",
    body: JSON.stringify({ class_id: classId, school_year: selectedAcademicYear || data.school.year }),
  });
  $("#dialogTitle").textContent = prepared.mode === "test" ? "Aperçu WhatsApp · mode test" : "Liste prête pour WhatsApp";
  $("#dialogKicker").textContent = `${prepared.class_name} · ${prepared.school_year} · ${prepared.student_count} élèves`;
  $("#dialogFields").innerHTML = `<label class="field full"><span class="field-label">Message préparé pour ${esc(prepared.teacher_name)} · aucun envoi automatique</span><textarea class="field-control" readonly rows="14">${esc(prepared.message)}</textarea></label><p class="small-print muted">${prepared.mode === "test" ? "Mode test actif : aucun lien externe ni message n’a été envoyé. Le contenu est seulement affiché et la préparation est inscrite au journal d’audit." : "Le lien ouvre WhatsApp avec le numéro et le texte prérempli. Vérifiez le destinataire puis appuyez vous-même sur Envoyer dans WhatsApp."}</p>${prepared.url ? `<a class="button button-primary" href="${esc(prepared.url)}" target="_blank" rel="noopener noreferrer">Ouvrir WhatsApp</a>` : ""}<button class="button button-secondary" type="button" data-copy-whatsapp>Copier le texte</button><button class="button button-quiet" type="button" data-close-dialog>Fermer</button>`;
  $("#dialogError").hidden = true;
  $("#recordDialog").showModal();
}
function saveSettings() {
  if (!requirePermission("settings", "update")) return;
  let curriculumSubjects;
  try {
    curriculumSubjects = readCurriculumEditor();
  } catch (error) {
    showToast(error.message);
    return;
  }
  const name = $('[name="school-name"]')?.value.trim();
  if (!name) {
    showToast("Le nom de l’établissement est obligatoire.");
    return;
  }
  data.school.appName = $('[name="app-name"]')?.value.trim() || "Gestion scolaire";
  data.school.name = name;
  data.school.shortName = $('[name="school-short"]')?.value.trim() || name;
  const threshold = Number($('[name="promotion-threshold"]')?.value);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 20) {
    showToast("Le seuil de passage doit être compris entre 0 et 20.");
    return;
  }
  const countryCode = $('[name="country-calling-code"]')?.value.trim() || "223";
  if (!/^\d{1,4}$/.test(countryCode)) {
    showToast("L’indicatif téléphonique doit contenir entre 1 et 4 chiffres.");
    return;
  }
  data.school.promotionThreshold = threshold;
  data.school.curriculum[selectedCurriculumLevel] = curriculumSubjects;
  const conductMaximum = Number($('[name="conduct-maximum"]')?.value);
  if (![18, 20].includes(conductMaximum)) {
    showToast("La note maximale de conduite doit être de 18 ou 20 points.");
    return;
  }
  data.school.conductMaximum = conductMaximum;
  data.school.whatsappMode = $('[name="whatsapp-mode"]')?.value === "link" ? "link" : "test";
  data.school.countryCallingCode = countryCode;
  data.school.cardTemplate = {
    layout: $('[name="card-layout"]')?.value === "vertical" ? "vertical" : "horizontal",
    showPhoto: $('[name="card-showPhoto"]')?.checked ?? true,
    showMatricule: $('[name="card-showMatricule"]')?.checked ?? true,
    showBirthDate: $('[name="card-showBirthDate"]')?.checked ?? true,
    showGuardian: $('[name="card-showGuardian"]')?.checked ?? true,
    showFamilyPhone: $('[name="card-showFamilyPhone"]')?.checked ?? true,
  };
  data.school.address = $('[name="school-address"]')?.value.trim();
  data.school.phone = $('[name="school-phone"]')?.value.trim();
  data.school.email = $('[name="school-email"]')?.value.trim();
  data.school.color = $("#schoolColorInput")?.value || data.school.color;
  data.school.secondaryColor = $("#schoolSecondaryInput")?.value || data.school.secondaryColor;
  data.school.accentColor = $("#schoolAccentInput")?.value || data.school.accentColor;
  persist();
  renderPage();
  showToast("Identité de l’établissement enregistrée.");
}
async function saveCredentials() {
  const user = currentUser();
  if (!user || !["proviseur", "admin"].includes(user.role)) {
    showToast("Seul le Proviseur ou un administrateur peut modifier ses identifiants.");
    return;
  }
  const username = $("#adminUsername").value.trim();
  const currentPassword = $("#currentPassword").value;
  const newPassword = $("#newPassword").value;
  if (!username) { showToast("Le nom d’utilisateur ne peut pas être vide."); return; }
  if (newPassword && newPassword.length < 8) { showToast("Le nouveau mot de passe doit contenir au moins 8 caractères."); return; }
  if (!apiMode && data.users.some((item) => item.id !== user.id && item.username.toLowerCase() === username.toLowerCase())) {
    showToast("Ce nom d’utilisateur est déjà utilisé.");
    return;
  }
  if (apiMode) {
    try {
      const updated = await apiRequest("/api/credentials", {
        method: "PUT",
        body: JSON.stringify({ current_password: currentPassword, username, new_password: newPassword || null }),
      });
      sessionUser = updated;
      currentUserId = updated.id;
      data.users = data.users.map((item) => item.id === updated.id ? updated : item);
    } catch (error) {
      showToast(error.message || "Impossible de modifier les identifiants.");
      return;
    }
  } else {
    if (!(await verifyPassword(currentPassword, user))) {
      showToast("Le mot de passe actuel est incorrect.");
      return;
    }
    user.username = username;
    if (newPassword) Object.assign(user, await passwordDigest(newPassword));
    persist();
  }
  $("#currentPassword").value = "";
  $("#newPassword").value = "";
  renderPage();
  showToast("Identifiants administrateur mis à jour.");
}
async function downloadBackup() {
  let backup;
  if (apiMode) {
    backup = await apiRequest("/api/backup", { cache: "no-store" });
  } else {
    const { users: ignoredUsers, ...state } = structuredClone(data);
    backup = { format: "school-management-backup-v1", created_at: new Date().toISOString(), state };
  }
  const blob = new Blob([JSON.stringify(backup, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `sauvegarde-scolaire-${today()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  showToast("Sauvegarde téléchargée sans comptes ni mots de passe.");
}
async function refreshAutomationStatus() {
  const status = $("#automaticBackupStatus");
  if (!status || !apiMode) return;
  try {
    const result = await apiRequest("/api/automation", { cache: "no-store" });
    const latest = result.latest_backup
      ? `${result.latest_backup.filename} · ${new Intl.DateTimeFormat("fr-FR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(result.latest_backup.created_at))}`
      : "Aucune sauvegarde créée";
    status.textContent = `${result.automatic_backup_enabled ? `Automatique toutes les ${result.interval_hours} h` : "Automatique désactivée"} · ${result.backup_count}/${result.retention} sauvegardes conservées · Dernière : ${latest}${result.last_error ? ` · ${result.last_error}` : ""}`;
  } catch (error) {
    status.textContent = error.message || "État des sauvegardes indisponible.";
  }
}
async function createAutomaticBackup() {
  if (!requirePermission("settings", "update")) return;
  try {
    const result = await apiRequest("/api/automation/backup", { method: "POST" });
    await refreshAutomationStatus();
    showToast(`Sauvegarde locale créée : ${result.backup.filename}`);
  } catch (error) {
    showToast(error.message || "Impossible de créer la sauvegarde locale.");
  }
}
async function downloadFinancialPdf() {
  if (!requirePermission("accounting", "read")) return;
  const query = new URLSearchParams();
  if (selectedAccountingPeriod.start) query.set("from_date", selectedAccountingPeriod.start);
  if (selectedAccountingPeriod.end) query.set("to_date", selectedAccountingPeriod.end);
  const response = await fetch(`/api/reports/financial-summary.pdf${query.size ? `?${query}` : ""}`, {
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = "Impossible de générer le rapport PDF.";
    try { detail = (await response.json()).detail || detail; } catch { /* The server may return a non-JSON error. */ }
    throw new Error(detail);
  }
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "rapport-financier-scolaire.pdf";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  showToast("Rapport financier PDF généré par le serveur Python.");
}
async function restoreBackup(file) {
  if (!file || file.size > MAX_BACKUP_BYTES) throw new Error("Le fichier de sauvegarde dépasse 64 Mo.");
  let backup;
  try {
    backup = JSON.parse(await file.text());
  } catch {
    throw new Error("Le fichier n’est pas un JSON valide.");
  }
  if (backup?.format !== "school-management-backup-v1" || !backup.state || typeof backup.state !== "object" || Array.isArray(backup.state)) {
    throw new Error("Ce fichier n’est pas une sauvegarde scolaire reconnue.");
  }
  if ("users" in backup.state) throw new Error("Une sauvegarde scolaire ne doit pas contenir de comptes.");
  const required = ["students", "classes", "teachers", "payments", "expenses", "grades", "assessments", "incidents"];
  if (!backup.state.school || typeof backup.state.school !== "object" || required.some((key) => !Array.isArray(backup.state[key]))) {
    throw new Error("La sauvegarde est incomplète : établissement ou registres scolaires manquants.");
  }
  const restoredState = structuredClone(backup.state);
  restoredState.attendance ||= [];
  if (!Array.isArray(restoredState.attendance)) throw new Error("Le registre des présences est invalide.");
  if (!window.confirm("Restaurer cette sauvegarde et remplacer les données scolaires actuelles ? Les comptes, mots de passe et le journal d’audit seront conservés.")) return;
  const users = data.users;
  if (apiMode) {
    await apiRequest("/api/backup", { method: "POST", body: JSON.stringify({ format: backup.format, state: restoredState }) });
    const result = await apiRequest("/api/state", { cache: "no-store" });
    if (!result.state) throw new Error("La sauvegarde a été restaurée, mais les données ne peuvent pas être rechargées.");
    data = { ...result.state, analytics: result.analytics, users };
  } else {
    data = { ...restoredState, users };
    ensureCollections();
    persist();
  }
  ensureCollections();
  renderPage();
  showToast("Sauvegarde restaurée. Les comptes et l’historique d’audit ont été conservés.");
}
function saveCurriculum() {
  if (!requirePermission("curriculum", "update")) return;
  try {
    data.school.curriculum[selectedCurriculumLevel] = readCurriculumEditor();
  } catch (error) {
    showToast(error.message);
    return;
  }
  persist();
  renderPage();
  showToast(`Référentiel de ${academicLabel(selectedCurriculumLevel)} enregistré.`);
}
async function saveAttendance() {
  if (!can("attendance", "create") && !can("attendance", "update")) {
    showToast("Cette action n’est pas autorisée pour votre compte.");
    return;
  }
  if (!selectedAttendanceClass || !/^\d{4}-\d{2}-\d{2}$/.test(selectedAttendanceDate)) {
    showToast("Choisissez une classe et une date valides.");
    return;
  }
  const roster = studentsInClass(selectedAttendanceClass).filter((student) => student.active);
  let changed = 0;
  for (const student of roster) {
    const status = $(`[data-attendance-status="${CSS.escape(student.id)}"]`)?.value || "";
    const observation = $(`[data-attendance-observation="${CSS.escape(student.id)}"]`)?.value.trim() || "";
    const existing = data.attendance.find((item) => item.studentId === student.id && item.date === selectedAttendanceDate);
    if (!status) {
      if (existing && can("attendance", "delete")) {
        data.attendance = data.attendance.filter((item) => item.id !== existing.id);
        changed += 1;
      }
      continue;
    }
    if (existing && !can("attendance", "update")) continue;
    if (!existing && !can("attendance", "create")) continue;
    const entry = existing || { id: uid(), studentId: student.id, date: selectedAttendanceDate, className: selectedAttendanceClass, schoolYear: data.school.year, recordedBy: currentUser()?.username || "" };
    Object.assign(entry, { status, observation });
    if (!existing) data.attendance.push(entry);
    changed += 1;
  }
  if (!changed) {
    showToast("Aucune présence à enregistrer.");
    return;
  }
  persist();
  renderPage();
  showToast(`${changed} présence${changed > 1 ? "s" : ""} enregistrée${changed > 1 ? "s" : ""}.`);
}
async function handleClick(event) {
  const pageButton = event.target.closest("[data-page]");
  if (pageButton) { navigate(pageButton.dataset.page); return; }
  const pageLink = event.target.closest("[data-page-link]");
  if (pageLink) { navigate(pageLink.dataset.pageLink); return; }
  const actionButton = event.target.closest("[data-action]");
  if (!actionButton) return;
  const { action, id } = actionButton.dataset;
  switch (action) {
    case "new-student": openRecord("student"); break;
    case "edit-student": openRecord("student", id); break;
    case "delete-student": deleteRecord("student", id); break;
    case "new-class": openRecord("class"); break;
    case "edit-class": openRecord("class", id); break;
    case "delete-class": deleteRecord("class", id); break;
    case "class-students": navigate("eleves"); setTimeout(() => { $('[data-filter="student-class"]').value = data.classes.find((item) => item.id === id)?.name || ""; $('[data-filter="student-class"]').dispatchEvent(new Event("change")); handleSearch({ target: $('[data-search="students"]') }); }, 0); break;
    case "print-class-roster": printClassRoster(id); break;
    case "whatsapp-roster":
      try { await prepareWhatsAppRoster(id); }
      catch (error) { showToast(error.message || "Impossible de préparer la liste WhatsApp."); }
      break;
    case "close-academic-year":
      try { await closeAcademicYear(); }
      catch (error) { showToast(error.message || "Impossible de clôturer l’année scolaire."); }
      break;
    case "new-payment": openRecord("payment"); break;
    case "edit-payment": openRecord("payment", id); break;
    case "delete-payment": deleteRecord("payment", id); break;
    case "new-expense": openRecord("expense"); break;
    case "edit-expense": openRecord("expense", id); break;
    case "delete-expense": deleteRecord("expense", id); break;
    case "new-grade": openRecord("grade"); break;
    case "delete-grade": deleteRecord("grade", id); break;
    case "new-assessment": openRecord("assessment"); break;
    case "select-assessment": selectedAssessmentId = id; renderPage(); break;
    case "delete-assessment": deleteRecord("assessment", id); selectedAssessmentId = null; renderPage(); break;
    case "new-teacher": openRecord("teacher"); break;
    case "edit-teacher": openRecord("teacher", id); break;
    case "delete-teacher": deleteRecord("teacher", id); break;
    case "view-teacher":
      navigate("enseignants");
      if (can("teachers", "update")) openRecord("teacher", id);
      break;
    case "view-student": selectedDossierId = id; navigate("dossiers"); break;
    case "print-dossier": window.print(); break;
    case "student-payment":
      openRecord("payment");
      if ($("#recordDialog").open) $("#dialogFields [name='studentId']").value = id;
      break;
    case "new-account": openRecord("account"); break;
    case "edit-account": openRecord("account", id); break;
    case "delete-account": deleteRecord("account", id); break;
    case "toggle-account": toggleAccount(id); break;
    case "print-report": printReport(id); break;
    case "print-evaluation-history": printEvaluationHistory(id); break;
    case "new-incident": openRecord("incident", id); break;
    case "validate-incident": await validateIncident(id); break;
    case "delete-incident": deleteRecord("incident", id); break;
    case "clear-accounting-period":
      selectedAccountingPeriod = { start: "", end: "" };
      try {
        await refreshAnalytics();
      } catch (error) {
        showToast(error.message || "Impossible de recalculer le rapport.");
        break;
      }
      renderPage();
      break;
    case "add-curriculum-subject":
      if (can("curriculum", "update") || can("settings", "update")) {
        data.school.curriculum[selectedCurriculumLevel] ||= [];
        data.school.curriculum[selectedCurriculumLevel].push({ name: "", coefficient: 1, optional: false });
        renderPage();
        $(`[data-curriculum-subject="${data.school.curriculum[selectedCurriculumLevel].length - 1}"]`)?.focus();
      }
      break;
    case "remove-curriculum-subject":
      if (can("curriculum", "update") || can("settings", "update")) {
        data.school.curriculum[selectedCurriculumLevel].splice(Number(actionButton.dataset.index), 1);
        renderPage();
      }
      break;
    case "print-class-list": {
      if (!canRead("students")) { showToast("Votre compte ne peut pas consulter les listes d’élèves."); break; }
      const student = studentById($("#gradeStudent")?.value);
      if (student) { navigate("eleves"); $('[data-filter="student-class"]').value = student.className; handleSearch({ target: $('[data-search="students"]') }); window.print(); }
      break;
    }
    case "print": window.print(); break;
    case "save-settings": saveSettings(); break;
    case "save-curriculum": saveCurriculum(); break;
    case "save-attendance": await saveAttendance(); break;
    case "save-credentials": await saveCredentials(); break;
    case "remove-logo":
      if (requirePermission("settings", "update")) { data.school.logo = ""; persist(); renderPage(); showToast("Logo retiré."); }
      break;
    case "export-data": {
      try { await downloadBackup(); }
      catch (error) { showToast(error.message || "Impossible de créer la sauvegarde."); }
      break;
    }
    case "create-automatic-backup":
      await createAutomaticBackup();
      break;
    case "download-financial-pdf":
      try { await downloadFinancialPdf(); }
      catch (error) { showToast(error.message || "Impossible de générer le rapport PDF."); }
      break;
    case "restore-data":
      $("#backupFileInput")?.click();
      break;
    case "reset-data":
      if (currentUser()?.role === "proviseur" && window.confirm("Effacer les dossiers scolaires et recharger les données d’exemple ? Les comptes et le journal d’audit seront conservés.")) {
        if (apiMode) {
          try {
            await apiRequest("/api/reset-demo", { method: "POST" });
            const users = data.users;
            data = exampleData();
            data.users = users;
            persist();
            renderPage();
            showToast("Les données d’exemple ont été restaurées.");
          } catch (error) {
            showToast(error.message || "Impossible de réinitialiser les données.");
          }
        } else {
          localStorage.removeItem(STORAGE_KEY);
          await initialize();
          showToast("Les données d’exemple ont été restaurées.");
        }
      }
      break;
  }
}
function applyBranding() {
  document.title = `${data.school.appName || "Gestion scolaire"} · ${data.school.name}`;
  document.documentElement.style.setProperty("--school-color", data.school.color || "#183a4d");
  document.documentElement.style.setProperty("--school-color-secondary", data.school.secondaryColor || "#dce8e4");
  document.documentElement.style.setProperty("--school-color-accent", data.school.accentColor || "#c57b53");
  $(".brand-copy span").textContent = (data.school.appName || "Gestion scolaire").toLocaleUpperCase("fr");
  const logo = $("#sidebarLogo");
  logo.innerHTML = data.school.logo ? `<img src="${esc(data.school.logo)}" alt="">` : esc(initials(data.school.shortName || data.school.name));
}
function login(user) {
  currentUserId = user.id;
  sessionUser = user;
  $("#loginScreen").hidden = true;
  $("#appShell").hidden = false;
  applyBranding();
  const firstAvailablePage = Object.entries({ dashboard: "dashboard", students: "eleves", classes: "classes", attendance: "presences", curriculum: "matieres", dossiers: "dossiers", teachers: "enseignants", discipline: "discipline", payments: "paiements", expenses: "depenses", accounting: "comptabilite", grades: "bulletins", cards: "cartes", accounts: "comptes", audit: "audit", settings: "reglages" }).find(([module]) => canRead(module));
  activePage = canRead("dashboard") ? "dashboard" : firstAvailablePage?.[1] || "dashboard";
  renderPage();
}
async function logout() {
  if (apiMode && sessionUser) {
    try { await apiRequest("/api/logout", { method: "POST" }); }
    catch (error) {
      console.error("La session locale n’a pas pu être fermée côté serveur.", error);
      showToast(error.message || "Déconnexion impossible. Réessayez.");
      return;
    }
  }
  currentUserId = null;
  sessionUser = null;
  $("#appShell").hidden = true;
  $("#loginScreen").hidden = false;
  $("#loginPassword").value = "";
  $("#loginUsername").value = "";
  $("#loginUsername").focus();
}
async function initialize() {
  try {
    const response = await fetch("/api/bootstrap", { credentials: "same-origin", cache: "no-store" });
    if (response.ok) {
      apiMode = true;
      $("#runtimeModeLabel").innerHTML = "Serveur local<br><small>SQLite · permissions vérifiées</small>";
      $(".login-footnote").textContent = "Serveur local · Comptes et autorisations vérifiés côté serveur";
      const bootstrap = await response.json();
      data = exampleData();
      if (bootstrap.user) {
        sessionUser = bootstrap.user;
        currentUserId = bootstrap.user.id;
        data = bootstrap.state || exampleData();
        data.users = [bootstrap.user];
        if (bootstrap.user.permissions?.accounts?.read) {
          const accounts = await apiRequest("/api/accounts");
          data.users = accounts.users;
        }
        ensureCollections();
        $("#loginSchoolName").textContent = data.school.appName || data.school.name;
        $(".scene-topline strong").textContent = data.school.year;
        login(bootstrap.user);
        if (!bootstrap.state && bootstrap.user.role === "proviseur") persist();
      } else {
        $("#loginSchoolName").textContent = data.school.appName || data.school.name;
        $(".scene-topline strong").textContent = data.school.year;
      }
      return;
    }
    if (response.status !== 404) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Impossible d’initialiser l’application (${response.status}).`);
    }
  } catch (error) {
    if (error instanceof TypeError) {
      apiMode = false;
    } else {
      $("#loginError").textContent = error.message || "Le serveur scolaire ne répond pas.";
      $("#loginError").hidden = false;
      return;
    }
  }
  $("#runtimeModeLabel").innerHTML = "Mode démonstration<br><small>Données propres à ce navigateur</small>";
  $(".login-footnote").textContent = "Mode démonstration · Données et droits conservés dans ce navigateur";
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    try { data = JSON.parse(saved); } catch (error) {
      console.error("Les données scolaires enregistrées ne sont pas lisibles.", error);
      $("#loginError").textContent = "Les données locales sont illisibles. Exportez-les depuis une sauvegarde ou réinitialisez la démonstration.";
      $("#loginError").hidden = false;
      return;
    }
  } else {
    data = exampleData();
    const credentials = await passwordDigest(FIRST_PASSWORD);
    data.users = [{ id: uid(), username: "Moussa", role: "proviseur", active: true, permissions: permissionsForRole("proviseur"), ...credentials }];
    persist();
  }
  if (!data.school || !Array.isArray(data.users) || !Array.isArray(data.students)) {
    $("#loginError").textContent = "Les données de démonstration sont incomplètes. Réinitialisez-les pour continuer.";
    $("#loginError").hidden = false;
    return;
  }
  ensureCollections();
  persist();
  $("#loginSchoolName").textContent = data.school.appName || data.school.name;
  $(".scene-topline strong").textContent = data.school.year;
}
function ensureCollections() {
  const examples = exampleData();
  data.school ||= examples.school;
  data.school.appName ||= examples.school.appName;
  data.school.secondaryColor ||= examples.school.secondaryColor;
  data.school.accentColor ||= examples.school.accentColor;
  normalizeAcademicCatalog(data.school);
  data.school.promotionThreshold ??= 10;
  data.school.conductMaximum ??= 18;
  data.school.whatsappMode ||= "test";
  data.school.countryCallingCode ||= "223";
  data.school.academicClosures ||= [];
  data.school.cardTemplate ||= { ...DEFAULT_CARD_TEMPLATE };
  data.school.cardTemplate = { ...DEFAULT_CARD_TEMPLATE, ...data.school.cardTemplate };
  data.users ||= [];
  data.teachers ||= examples.teachers;
  data.assessments ||= [];
  data.incidents ||= [];
  data.attendance ||= [];
  data.grades ||= [];
  data.expenses ||= [];
  data.payments ||= [];
  data.classes ||= [];
  data.students ||= [];
  data.users.forEach((user) => {
    const defaults = permissionsForRole(user.role);
    user.permissions ||= {};
    for (const [module, actions] of Object.entries(defaults)) {
      user.permissions[module] ||= {};
      for (const [action, allowed] of Object.entries(actions)) {
        user.permissions[module][action] ??= allowed;
      }
    }
    user.cycle_scope ||= valuesForDirectorRole(user.role);
  });
  data.classes.forEach((schoolClass) => {
    schoolClass.academicLevel ||= LEGACY_CLASS_LEVELS[schoolClass.name] || "";
    schoolClass.academicLevel = ACADEMIC_LEVEL_ALIASES[schoolClass.academicLevel] || schoolClass.academicLevel;
  });
  data.students.forEach((student) => {
    student.matricule ||= nextMatricule();
    student.enrollmentYear ||= data.school.year;
    student.enrollmentHistory ||= [{ schoolYear: student.enrollmentYear, className: student.className, decision: "inscrit" }];
  });
  data.assessments.forEach((assessment) => { assessment.schoolYear ||= data.school.year; });
  data.grades.forEach((grade) => { grade.schoolYear ||= data.assessments.find((item) => item.id === grade.assessmentId)?.schoolYear || data.school.year; });
}

$("#loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const username = $("#loginUsername").value.trim();
  const password = $("#loginPassword").value;
  if (apiMode) {
    try {
      const result = await apiRequest("/api/login", { method: "POST", body: JSON.stringify({ username, password }) });
      sessionUser = result.user;
      currentUserId = result.user.id;
      const statePayload = await apiRequest("/api/state");
      data = statePayload.state || exampleData();
      data.users = [result.user];
      if (result.user.permissions?.accounts?.read) {
        const accounts = await apiRequest("/api/accounts");
        data.users = accounts.users;
      }
      ensureCollections();
      if (!statePayload.state) persist();
      $("#loginError").hidden = true;
      $("#loginPassword").value = "";
      login(result.user);
    } catch (error) {
      sessionUser = null;
      currentUserId = null;
      $("#loginError").textContent = error.message || "Nom d’utilisateur ou mot de passe incorrect.";
      $("#loginError").hidden = false;
      $("#loginPassword").value = "";
    }
    return;
  }
  const user = data.users.find((item) => item.username.toLowerCase() === username.toLowerCase());
  if (!user || !user.active || !(await verifyPassword(password, user))) {
    $("#loginError").textContent = user?.active === false ? "Ce compte est désactivé. Contactez l’administrateur." : "Nom d’utilisateur ou mot de passe incorrect.";
    $("#loginError").hidden = false;
    return;
  }
  $("#loginError").hidden = true;
  $("#loginPassword").value = "";
  login(user);
});
$("#logoutButton").addEventListener("click", logout);
$("#sidebar").addEventListener("click", handleClick);
$("#pageContent").addEventListener("click", handleClick);
$("#pageContent").addEventListener("input", handleSearch);
$("#pageContent").addEventListener("input", (event) => {
  const target = event.target;
  if (target.matches("[data-curriculum-subject], [data-curriculum-coefficient]")) syncCurriculumDraft();
  if (target.matches("#schoolColorInput, #schoolSecondaryInput, #schoolAccentInput")) {
    const property = target.id === "schoolColorInput" ? "--school-color" : target.id === "schoolSecondaryInput" ? "--school-color-secondary" : "--school-color-accent";
    document.documentElement.style.setProperty(property, target.value);
    target.closest(".field")?.querySelector(".color-swatch-preview")?.style.setProperty("background", target.value);
    const preview = $("#brandPreview");
    if (preview) {
      preview.style.background = $("#schoolColorInput")?.value || data.school.color;
      preview.style.borderColor = $("#schoolAccentInput")?.value || data.school.accentColor;
      preview.querySelector("strong").textContent = $('[name="app-name"]')?.value || "Gestion scolaire";
      preview.querySelector("span").textContent = $('[name="school-name"]')?.value || data.school.name;
    }
  }
  if (target.matches('[name="app-name"], [name="school-name"]')) {
    const preview = $("#brandPreview");
    if (preview) {
      preview.querySelector("strong").textContent = $('[name="app-name"]')?.value || "Gestion scolaire";
      preview.querySelector("span").textContent = $('[name="school-name"]')?.value || data.school.name;
    }
  }
});
$("#pageContent").addEventListener("change", async (event) => {
  if (event.target.matches("#attendanceClass")) {
    selectedAttendanceClass = event.target.value;
    renderPage();
  }
  if (event.target.matches("#attendanceDate")) {
    selectedAttendanceDate = event.target.value;
    renderPage();
  }
  if (event.target.matches("#curriculumLevel")) {
    try {
      data.school.curriculum[selectedCurriculumLevel] = readCurriculumEditor();
    } catch (error) {
      showToast(error.message);
      event.target.value = selectedCurriculumLevel;
      return;
    }
    selectedCurriculumLevel = event.target.value;
    renderPage();
  }
  if (event.target.matches("[data-curriculum-optional]")) syncCurriculumDraft();
  if (event.target.matches("[data-card-template]")) {
    const preview = $("#cardTemplatePreview");
    if (preview) {
      preview.classList.toggle("vertical", $('[name="card-layout"]')?.value === "vertical");
      $$("#cardTemplatePreview [data-preview-field]").forEach((item) => {
        const key = item.dataset.previewField;
        const checkbox = $(`[name="card-${key === "photo" ? "showPhoto" : key === "matricule" ? "showMatricule" : key === "birthDate" ? "showBirthDate" : key === "guardian" ? "showGuardian" : key === "familyPhone" ? "showFamilyPhone" : ""}]`);
        item.hidden = Boolean(checkbox && !checkbox.checked);
      });
    }
  }
  if (event.target.matches("#classAcademicYear")) {
    selectedAcademicYear = event.target.value;
    promotionDraft = {};
    renderPage();
  }
  if (event.target.matches("[data-year-decision]")) {
    const id = event.target.dataset.yearDecision;
    promotionDraft[id] ||= {};
    promotionDraft[id].decision = event.target.value;
    renderPage();
  }
  if (event.target.matches("[data-year-target]")) {
    const id = event.target.dataset.yearTarget;
    promotionDraft[id] ||= {};
    promotionDraft[id].targetClassId = event.target.value;
  }
  if (event.target.matches('[data-filter="student-class"]')) handleSearch({ target: $('[data-search="students"]') });
  if (event.target.matches("#accountSelect")) renderPage();
  if (event.target.matches("#dossierStudent")) {
    selectedDossierId = event.target.value;
    renderPage();
  }
  if (event.target.matches("#gradeStudent")) {
    selectedGradeStudentId = event.target.value;
    renderPage();
  }
  if (event.target.matches("#evaluationYear")) {
    selectedEvaluationYear = event.target.value;
    selectedAssessmentId = null;
    renderPage();
  }
  if (event.target.matches("#accountingFrom, #accountingTo")) {
    const next = {
      start: $("#accountingFrom")?.value || "",
      end: $("#accountingTo")?.value || "",
    };
    if (next.start && next.end && next.start > next.end) {
      showToast("La date de début doit précéder la date de fin.");
      renderPage();
      return;
    }
    selectedAccountingPeriod = next;
    try {
      await refreshAnalytics(next.start, next.end);
    } catch (error) {
      showToast(error.message || "Impossible de recalculer le rapport pour cette période.");
      return;
    }
    renderPage();
  }
  if (event.target.matches("#gradeTerm")) {
    selectedGradeTerm = event.target.value;
    renderPage();
  }
  if (event.target.matches("#schoolLogoInput") && event.target.files?.[0]) {
    if (!requirePermission("settings", "update")) return;
    const file = event.target.files[0];
    if (!["image/png", "image/jpeg", "image/svg+xml"].includes(file.type) || file.size > 1_500_000) {
      showToast("Choisissez un logo PNG, JPG ou SVG de moins de 1,5 Mo.");
      event.target.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => showToast("Impossible de lire ce fichier image.");
    reader.onload = () => { data.school.logo = reader.result; persist(); renderPage(); showToast("Logo enregistré."); };
    reader.readAsDataURL(file);
  }
  if (event.target.matches("#backupFileInput") && event.target.files?.[0]) {
    restoreBackup(event.target.files[0]).catch((error) => showToast(error.message || "Impossible de restaurer cette sauvegarde."));
    event.target.value = "";
  }
});
$("#pageContent").addEventListener("change", updatePermission);
$("#pageContent").addEventListener("change", (event) => {
  const input = event.target.closest("[data-grade-score], [data-assessment-score]");
  if (!input) return;
  if (!requirePermission("grades", "update")) return;
  const score = input.value === "" ? null : Number(input.value);
  if (score !== null && (!Number.isFinite(score) || score < 0 || score > 20)) {
    showToast("La note doit être comprise entre 0 et 20.");
    renderPage();
    return;
  }
  const grade = data.grades.find((item) => item.id === (input.dataset.gradeScore || input.dataset.assessmentScore));
  if (!grade) return;
  grade.score = score;
  persist();
  const assessmentId = selectedAssessmentId;
  const studentId = selectedGradeStudentId;
  const term = selectedGradeTerm;
  renderPage();
  selectedAssessmentId = assessmentId;
  selectedGradeStudentId = studentId;
  selectedGradeTerm = term;
  showToast("Note enregistrée. La moyenne a été recalculée.");
});
$("#recordDialog").addEventListener("change", (event) => {
  if (event.target.matches("#studentPhotoInput")) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/") || file.size > 1_000_000) {
      showToast("La photo doit être une image de moins de 1 Mo.");
      event.target.value = "";
      return;
    }
    prepareStudentPhoto(file).then((photo) => {
      dialogPhoto = photo;
      $("#studentPhotoPreview").innerHTML = `<img src="${photo}" alt="Aperçu de la photo">`;
    }).catch((error) => {
      event.target.value = "";
      showToast(error.message || "Impossible de lire ou compresser la photo.");
    });
    return;
  }
  const studentId = event.target.matches('#dialogFields [name="studentId"]') ? event.target.value : null;
  const className = event.target.matches('#dialogFields [name="className"]') ? event.target.value : null;
  const subjectChanged = event.target.matches('#dialogFields [name="subject"]');
  const isGradeStudent = dialogState?.kind === "grade" && studentId;
  const isAssessmentClass = dialogState?.kind === "assessment" && className;
  if (!isGradeStudent && !isAssessmentClass && !(dialogState?.kind === "grade" && subjectChanged)) return;
  const studentClassName = isGradeStudent ? data.students.find((student) => student.id === studentId)?.className : className;
  const subjectSelect = $('#dialogFields [name="subject"]');
  if (!subjectChanged) {
    const choices = subjectChoicesForClass(studentClassName);
    const previous = subjectSelect.value;
    subjectSelect.innerHTML = choices.map((choice) => `<option value="${esc(choice.value)}">${esc(choice.label)}</option>`).join("");
    subjectSelect.value = choices.some((choice) => choice.value === previous) ? previous : (choices[0]?.value || "");
  }
  if (dialogState.kind === "grade") {
    const coefficient = $('#dialogFields [name="coefficient"]');
    const selectedClass = isGradeStudent ? data.students.find((student) => student.id === studentId)?.className : data.students.find((student) => student.id === $('#dialogFields [name="studentId"]').value)?.className;
    if (coefficient) coefficient.value = coefficientForClassSubject(selectedClass, subjectSelect.value);
  }
});
$("#recordForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!dialogState || !requirePermission(dialogState.module, dialogState.id ? "update" : "create")) return;
  const values = Object.fromEntries(new FormData(event.currentTarget));
  const classScope = event.currentTarget.elements.namedItem("class_scope");
  if (classScope) values.class_scope = [...classScope.selectedOptions].map((option) => option.value);
  const cycleScope = event.currentTarget.elements.namedItem("cycle_scope");
  if (cycleScope) values.cycle_scope = [...cycleScope.selectedOptions].map((option) => option.value);
  const classIds = event.currentTarget.elements.namedItem("classIds");
  if (classIds) values.classIds = [...classIds.selectedOptions].map((option) => option.value);
  try { await saveRecord(values); } catch (error) {
    $("#dialogError").textContent = error.message || "Impossible d’enregistrer ce dossier.";
    $("#dialogError").hidden = false;
  }
});
$("#recordDialog").addEventListener("change", (event) => {
  if (event.target.matches('#dialogFields [name="role"]')) {
    const classScope = $('#dialogFields [name="class_scope"]');
    if (classScope) classScope.required = event.target.value === "enseignant";
    syncCycleScopeOptions(event.target.value, []);
  }
});
$$("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => $("#recordDialog").close()));
$("#recordDialog").addEventListener("click", (event) => {
  if (event.target === $("#recordDialog")) $("#recordDialog").close();
  if (event.target.closest("[data-close-dialog]")) $("#recordDialog").close();
  if (event.target.closest("[data-copy-whatsapp]")) {
    const message = $('#dialogFields textarea[readonly]')?.value || "";
    const clipboard = navigator.clipboard;
    if (clipboard?.writeText) {
      clipboard.writeText(message).then(() => showToast("Liste copiée.")).catch(() => showToast("Copie impossible; sélectionnez le texte manuellement."));
    } else {
      const textarea = $('#dialogFields textarea[readonly]');
      textarea?.select();
      showToast(document.execCommand("copy") ? "Liste copiée." : "Sélectionnez le texte manuellement.");
    }
  }
});
$("#menuToggle").addEventListener("click", () => {
  $("#sidebar").classList.toggle("open");
  $("#sidebarScrim").classList.toggle("active");
});
$("#sidebarScrim").addEventListener("click", () => {
  $("#sidebar").classList.remove("open");
  $("#sidebarScrim").classList.remove("active");
});

initialize().catch((error) => {
  console.error("Échec de l’initialisation de Gestion scolaire.", error);
  $("#loginError").textContent = "Impossible de préparer les données locales. Ouvrez cette application via http://localhost.";
  $("#loginError").hidden = false;
});
