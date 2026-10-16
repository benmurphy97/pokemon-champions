// Run battles between two teams on Showdown's engine, in-process (no server).
//
//   node sim/run.js <teamA.txt> <teamB.txt> [--battles N] [--seed N] [--format ID] [--validate]
//                   [--agent-a random|greedy] [--agent-b random|greedy]
//
// Prints one JSON object to stdout: wins, turns and speed. Agents default to random legal moves;
// `greedy` is the damage-estimating agent in sim/agents/greedy.js.

const fs = require('fs');
const path = require('path');

const SHOWDOWN = path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim');
const {BattleStream, getPlayerStreams, Teams, TeamValidator} = require(SHOWDOWN);
const {RandomPlayerAI} = require(path.join(SHOWDOWN, 'tools', 'random-player-ai'));

const {GreedyPlayer} = require('./agents/greedy');

const AGENTS = ['random', 'greedy'];
const DEFAULT_FORMAT = 'gen9championsvgc2026regmc';

function parseArgs(argv) {
	const opts = {battles: 100, seed: 1, format: DEFAULT_FORMAT, validate: false, agentA: 'random', agentB: 'random', teams: []};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (a === '--battles') opts.battles = Number(argv[++i]);
		else if (a === '--seed') opts.seed = Number(argv[++i]);
		else if (a === '--format') opts.format = argv[++i];
		else if (a === '--validate') opts.validate = true;
		else if (a === '--agent-a') opts.agentA = argv[++i];
		else if (a === '--agent-b') opts.agentB = argv[++i];
		else opts.teams.push(a);
	}
	for (const agent of [opts.agentA, opts.agentB]) {
		if (!AGENTS.includes(agent)) throw new Error(`unknown agent '${agent}', expected one of ${AGENTS}`);
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

function makeAgent(kind, stream, seed, getBattle) {
	return kind === 'greedy' ? new GreedyPlayer(stream, {seed}, getBattle) : new RandomPlayerAI(stream, {seed});
}

async function playOne(format, sets1, sets2, seed, agents) {
	const battleStream = new BattleStream();
	const streams = getPlayerStreams(battleStream);
	const getBattle = () => battleStream.battle;
	const battleSeed = [seed, seed + 1, seed + 2, seed + 3];
	const ai1 = makeAgent(agents[0], streams.p1, battleSeed.map(n => n + 100), getBattle);
	const ai2 = makeAgent(agents[1], streams.p2, battleSeed.map(n => n + 200), getBattle);
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
	const result = {format: opts.format, agentA: opts.agentA, agentB: opts.agentB, battles: opts.battles, winsA: 0, winsB: 0, ties: 0, noResult: 0};
	let totalTurns = 0;
	for (let i = 0; i < opts.battles; i++) {
		const {winner, turns} = await playOne(opts.format, a, b, opts.seed + i * 10, [opts.agentA, opts.agentB]);
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
