// Play many specific matchups in-process and report each one's win rate.
//
//   node sim/matchups.js <matchups.json> [--games N] [--agent greedy|random] [--seed N] [--out file.json]
//
// matchups.json: [{id, a: "<team text>", b: "<team text>"}, ...]
// Output: [{id, winsA, games, error?}, ...]. A matchup with an illegal team is reported with
// an `error` and not played. Seats alternate between games so neither side is favoured.

const fs = require('fs');
const path = require('path');
const {Teams} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {playOne, validate, DEFAULT_FORMAT} = require('./run');

function parseArgs(argv) {
	const opts = {games: 20, agent: 'greedy', seed: 1, out: null, file: null};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === '--games') opts.games = Number(argv[++i]);
		else if (a === '--agent') opts.agent = argv[++i];
		else if (a === '--seed') opts.seed = Number(argv[++i]);
		else if (a === '--out') opts.out = argv[++i];
		else opts.file = a;
	}
	if (!opts.file) throw new Error('usage: node sim/matchups.js <matchups.json> [options]');
	return opts;
}

async function main() {
	const opts = parseArgs(process.argv.slice(2));
	const matchups = JSON.parse(fs.readFileSync(opts.file, 'utf8'));
	const results = [];
	const t0 = Date.now();

	for (const [index, m] of matchups.entries()) {
		const a = Teams.import(m.a), b = Teams.import(m.b);
		const problems = [...validate(DEFAULT_FORMAT, a), ...validate(DEFAULT_FORMAT, b)];
		if (problems.length) {
			results.push({id: m.id, error: problems[0]});
			continue;
		}
		let winsA = 0;
		for (let g = 0; g < opts.games; g++) {
			const swap = g % 2 === 1;
			const seed = opts.seed + index * 1000 + g * 10;
			const {winner} = await playOne(DEFAULT_FORMAT, swap ? b : a, swap ? a : b, seed, [opts.agent, opts.agent]);
			if ((winner === 'A') !== swap && winner !== null && winner !== 'tie') winsA++;
		}
		results.push({id: m.id, winsA, games: opts.games});
	}

	const seconds = (Date.now() - t0) / 1000;
	if (opts.out) fs.writeFileSync(opts.out, JSON.stringify(results));
	const played = results.filter(r => !r.error).length;
	console.log(JSON.stringify({matchups: matchups.length, played, errors: results.length - played,
		seconds: +seconds.toFixed(1)}));
}

main().catch(err => { console.error(err.message); process.exit(2); });
