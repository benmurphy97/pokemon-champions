// Check many teams against the format's rules at once.
//
//   node sim/validate_many.js <teams.json> <problems.json>
//
// teams.json: list of Showdown team texts. problems.json: for each team, the list of rule violations (empty = legal).

const fs = require('fs');
const {Teams} = require(require('path').join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {validate, DEFAULT_FORMAT} = require('./run');

const [input, output] = process.argv.slice(2);
const texts = JSON.parse(fs.readFileSync(input, 'utf8'));
const problems = texts.map(text => {
	try {
		return validate(DEFAULT_FORMAT, Teams.import(text));
	} catch (err) {
		return [String(err.message)];
	}
});
fs.writeFileSync(output, JSON.stringify(problems));
console.log(`${problems.filter(p => !p.length).length} of ${texts.length} teams legal`);
