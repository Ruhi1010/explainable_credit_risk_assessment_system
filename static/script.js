const $ = (id) => document.getElementById(id);
const form = $('form');
const dial = $('dial');
const needle = $('needle');
const jitter = $('jitter');
const result = $('result');
const submitBtn = $('submit');

const TEXT_FIELDS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'];
const SAMPLE = {
  person_age: 27, person_income: 54000, person_home_ownership: 'RENT',
  person_emp_length: 4, cb_person_cred_hist_length: 5, cb_person_default_on_file: 'N',
  loan_intent: 'EDUCATION', loan_amnt: 12000, loan_int_rate: 11.5, loan_grade: 'C',
};

/* ---------- Dial ---------- */
const setNeedle = (p) => { needle.style.transform = `rotate(${-90 + p * 180}deg)`; };

// Power-on sweep: the one motion that plays without user input.
setTimeout(() => setNeedle(1), 350);
setTimeout(() => setNeedle(0), 1500);

function drawZones(threshold) {
  const t = threshold * 100;
  $('zoneLow').style.strokeDasharray = `${t} 100`;
  $('zoneHigh').style.strokeDasharray = `0 ${t} ${100 - t} 100`;
  dial.classList.add('scored');
}

function countUp(target, ms = 1300) {
  const start = performance.now();
  const step = (now) => {
    const k = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - k, 3);
    $('value').textContent = (target * eased).toFixed(1) + '%';
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------- Loan share of income (calculated) ---------- */
const syncShare = () => {
  const income = parseFloat(form.person_income.value);
  const amount = parseFloat(form.loan_amnt.value);
  form.loan_percent_income.value = income > 0 && amount >= 0 ? (amount / income).toFixed(2) : '';
};
form.person_income.addEventListener('input', syncShare);
form.loan_amnt.addEventListener('input', syncShare);

/* ---------- Sample data ---------- */
$('sample').addEventListener('click', () => {
  for (const [name, value] of Object.entries(SAMPLE)) {
    const field = form.elements[name];
    if (field instanceof RadioNodeList) field.value = value;
    else field.value = value;
  }
  syncShare();
});

/* ---------- Submit ---------- */
function buildPayload() {
  const payload = {};
  for (const [key, value] of new FormData(form)) {
    payload[key] = TEXT_FIELDS.includes(key) ? value : Number(value);
  }
  return payload;
}

function setBusy(busy) {
  submitBtn.disabled = busy;
  submitBtn.textContent = busy ? 'Assessing…' : 'Assess risk';
  jitter.classList.toggle('busy', busy);
}

function showMessage(risk, verdict, detail) {
  result.dataset.risk = risk;
  $('verdict').textContent = verdict;
  $('detail').textContent = detail;
}

function showResult({ default_probability: p, threshold: t, Result: label }) {
  const high = label === 'High Risk';
  drawZones(t);
  setNeedle(p);
  countUp(p * 100);
  showMessage(
    high ? 'high' : 'low',
    label,
    `The cutoff is ${(t * 100).toFixed(1)}%. Applications at or above it are flagged as high risk.`
  );
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  setBusy(true);
  showMessage('idle', 'Assessing…', 'Reading the application.');
  $('value').textContent = '–';
  setNeedle(0.5);

  // Free hosting can sleep; tell the user why a first request is slow.
  const slow = setTimeout(
    () => showMessage('idle', 'Assessing…', 'The server is waking up. The first request can take up to a minute.'),
    4000
  );

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });
    if (!res.ok) {
      throw new Error(
        res.status === 422
          ? 'The server rejected one of the values. Check each field and try again.'
          : `The server returned an error (${res.status}). Try again in a moment.`
      );
    }
    const data = await res.json();
    clearTimeout(slow);
    setBusy(false);
    // Let the needle settle out of the wobble before it swings to the result.
    requestAnimationFrame(() => showResult(data));
  } catch (err) {
    clearTimeout(slow);
    setBusy(false);
    setNeedle(0);
    showMessage('error', 'No result', err.message || 'Could not reach the server. Check your connection and try again.');
  }

  if (window.matchMedia('(max-width: 900px)').matches) {
    result.scrollIntoView({ behavior: 'smooth' });
  }
});
