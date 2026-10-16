// Round-robin between many teams, in-process.
//
//   node sim/tournament.js <pool.json> [--games N] [--seed N] [--agent greedy|random] [--out results.json]
//
// pool.json is a list of {id, species, text} (see src/champions/meta.py). Every pair of teams
// plays N games, half with each team in seat A, using different seeds in each half so the two
// halves are independent samples. Prints a summary and optionally writes full results.

const fs = require('fs');
const path = require('path');
const {Teams} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {playOne, DEFAULT_FORMAT} = require('./run');

function parseArgs(argv) {
	const opts = {games: 20, seed: 1, agent: 'greedy', out: null, pool: null};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === '--games') opts.games = Number(argv[++i]);
		else if (a === '--seed') opts.seed = Number(argv[++i]);
		else if (a === '--agent') opts.agent = argv[++i];
		else if (a === '--out') opts.out = argv[++i];
		else opts.pool = a;
	}
	if (!opts.pool) throw new Error('usage: node sim/tournament.js <pool.json> [options]');
	return opts;
}

async function main() {
	const opts = parseArgs(process.argv.slice(2));
	const pool = JSON.parse(fs.readFileSync(opts.pool, 'utf8'));
	const teams = pool.map(t => ({...t, sets: Teams.import(t.text)}));
	const n = teams.length;
	const wins = Array.from({length: n}, () => new Array(n).fill(0)); // wins[i][j]: games team i won vs j
	const played = Array.from({length: n}, () => new Array(n).fill(0));

	const t0 = Date.now();
	let games = 0;
	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			for (let g = 0; g < opts.games; g++) {
				// Swap seats halfway through so neither team always gets side A.
				const swap = g >= opts.games / 2;
				const [a, b] = swap ? [j, i] : [i, j];
				const seed = opts.seed + (i * n + j) * 1000 + g * 10;
				const {winner} = await playOne(DEFAULT_FORMAT, teams[a].sets, teams[b].sets, seed, [opts.agent, opts.agent]);
				games++;
				played[i][j]++; played[j][i]++;
				if (winner === 'A') wins[a][a === i ? j : i]++;
				else if (winner === 'B') wins[b][b === i ? j : i]++;
			}
		}
	}
	const seconds = (Date.now() - t0) / 1000;

	const rows = teams.map((t, i) => {
		const w = wins[i].reduce((x, y) => x + y, 0);
		const p = played[i].reduce((x, y) => x + y, 0);
		return {id: t.id, species: t.species, wins: w, games: p, winRate: +(w / p).toFixed(4)};
	}).sort((x, y) => y.winRate - x.winRate);

	const result = {agent: opts.agent, teams: n, gamesPerPair: opts.games, totalGames: games,
		seconds: +seconds.toFixed(1), gamesPerMinute: Math.round(games / seconds * 60), rows,
		matrix: {ids: teams.map(t => t.id), wins, played}};
	if (opts.out) fs.writeFileSync(opts.out, JSON.stringify(result));
	console.log(JSON.stringify({...result, matrix: undefined, rows: rows.slice(0, 5)}, null, 1));
}

main().catch(err => { console.error(err.message); process.exit(2); });
