// Run battles between two teams on Showdown's engine, in-process (no server).
//
//   node sim/run.js <teamA.txt> <teamB.txt> [--battles N] [--seed N] [--format ID] [--validate]
//
// Prints one JSON object to stdout: wins, turns and speed. Players pick random legal moves;
// real agents arrive in Phase 2.

const fs = require('fs');
const path = require('path');

const SHOWDOWN = path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim');
const {BattleStream, getPlayerStreams, Teams, TeamValidator} = require(SHOWDOWN);
const {RandomPlayerAI} = require(path.join(SHOWDOWN, 'tools', 'random-player-ai'));

const DEFAULT_FORMAT = 'gen9championsvgc2026regmc';

function parseArgs(argv) {
	const opts = {battles: 100, seed: 1, format: DEFAULT_FORMAT, validate: false, teams: []};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === '--battles') opts.battles = Number(argv[++i]);
		else if (a === '--seed') opts.seed = Number(argv[++i]);
		else if (a === '--format') opts.format = argv[++i];
		else if (a === '--validate') opts.validate = true;
		else opts.teams.push(a);
	}
	if (opts.teams.length !== 2) throw new Error('usage: node sim/run.js <teamA.txt> <teamB.txt> [options]');
	return opts;
}

function loadTeam(file) {
	const sets = Teams.import(fs.readFileSync(file, 'utf8'));
	if (!sets || !sets.length) throw new Error(`could not parse team: ${file}`);
	return sets;
}

// Returns a list of problems, empty if the team is legal in this format.
function validate(format, sets) {
	return TeamValidator.get(format).validateTeam(sets) || [];
}

async function playOne(format, sets1, sets2, seed) {
	const streams = getPlayerStreams(new BattleStream());
	const battleSeed = [seed, seed + 1, seed + 2, seed + 3];
	const ai1 = new RandomPlayerAI(streams.p1, {seed: battleSeed.map(n => n + 100)});
	const ai2 = new RandomPlayerAI(streams.p2, {seed: battleSeed.map(n => n + 200)});
	void ai1.start();
	void ai2.start();

	void streams.omniscient.write(
		`>start ${JSON.stringify({formatid: format, seed: battleSeed})}\n` +
		`>player p1 ${JSON.stringify({name: 'A', team: Teams.pack(sets1)})}\n` +
		`>player p2 ${JSON.stringify({name: 'B', team: Teams.pack(sets2)})}`
	);

	let winner = null, turns = 0;
	for await (const chunk of streams.omniscient) {
		for (const line of chunk.split('\n')) {
			if (line.startsWith('|turn|')) turns = Number(line.slice(6));
			else if (line.startsWith('|win|')) winner = line.slice(5);
			else if (line === '|tie') winner = 'tie';
		}
	}
	return {winner, turns};
}

async function main() {
	const opts = parseArgs(process.argv.slice(2));
	const [a, b] = opts.teams.map(loadTeam);

	const problems = {A: validate(opts.format, a), B: validate(opts.format, b)};
	if (opts.validate || problems.A.length || problems.B.length) {
		console.log(JSON.stringify({format: opts.format, problems}, null, 2));
		if (problems.A.length || problems.B.length) process.exit(1);
		return;
	}

	const t0 = Date.now();
	const result = {format: opts.format, battles: opts.battles, winsA: 0, winsB: 0, ties: 0, noResult: 0};
	let totalTurns = 0;
	for (let i = 0; i < opts.battles; i++) {
		const {winner, turns} = await playOne(opts.format, a, b, opts.seed + i * 10);
		totalTurns += turns;
		if (winner === 'A') result.winsA++;
		else if (winner === 'B') result.winsB++;
		else if (winner === 'tie') result.ties++;
		else result.noResult++;
	}
	const seconds = (Date.now() - t0) / 1000;
	result.meanTurns = +(totalTurns / opts.battles).toFixed(2);
	result.seconds = +seconds.toFixed(2);
	result.battlesPerMinute = Math.round(opts.battles / seconds * 60);
	console.log(JSON.stringify(result, null, 2));
}

main().catch(err => { console.error(err.message); process.exit(2); });
