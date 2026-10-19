// One-turn look-ahead for a neural player: play each of our candidate joint choices against the opponent's
// likely joint replies in copies of the battle, and keep the choice with the best health margin afterwards.
//
//   const choice = search(battle, 'p1', [{choice: 'move 1 1, move 2', prior: 0.4}, ...]);
//
// ORACLE: the copy holds the opponent's real sets (items, moves, spreads), which a player cannot see. This is
// an upper bound on what search can add, not a fair player. The opponent's replies come from the greedy
// agent's ranking (top REPLIES per slot), and their own pending choice is cleared so it is never seen.

const path = require('path');
const {Battle} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {GreedyPlayer} = require('./agents/greedy');

const REPLIES = 3;          // opponent options kept per slot (so up to 9 joint replies)
const WORST_WEIGHT = 0.5;   // payoff = (1 - w) * mean over replies + w * worst reply
const PRIOR_WEIGHT = 0.02;  // a small pull towards the policy's own preference, to break near-ties

function healthLeft(side) {
	return side.pokemon.reduce((sum, p) => sum + (p.fainted ? 0 : p.hp / p.maxhp), 0);
}

// The opponent's likely joint choices, from the greedy agent's per-slot ranking.
function replies(battle, foe) {
	const request = foe.activeRequest;
	if (!request || request.wait || request.forceSwitch || request.teamPreview || !request.active) return null;
	const scorer = Object.create(GreedyPlayer.prototype);
	Object.assign(scorer, {getBattle: () => battle, blind: false, request, claimed: [0, 0]});
	const perSlot = request.active.map((active, slot) => {
		const mon = foe.active[slot];
		if (!mon || mon.fainted || request.side.pokemon[slot].commanding) return ['pass'];
		const moves = active.moves.map((move, i) => ({move: {...move, slot: i + 1}})).filter(m => !m.move.disabled);
		if (!moves.length) return ['move 1'];
		return scorer.rankMoves(slot, moves).slice(0, REPLIES).map(o => o.choice);
	});
	const mega = request.active.findIndex(a => a.canMegaEvo);
	const joint = [];
	for (const a of perSlot[0]) {
		for (const b of perSlot[1] || ['pass']) {
			const pair = [a, b].map((c, slot) => slot === mega && c.startsWith('move') ? c + ' mega' : c);
			joint.push(pair.slice(0, request.active.length).join(', '));
		}
	}
	return joint;
}

function search(battle, side, candidates) {
	const me = battle[side], foe = battle[side === 'p1' ? 'p2' : 'p1'];
	const theirs = replies(battle, foe);
	if (!theirs || candidates.length < 2) return candidates[0].choice;
	// toJSON shares arrays (the log among them) with the live battle: detach, or every copy writes into the real game.
	const snapshot = JSON.parse(JSON.stringify(battle.toJSON()));
	const seed = [0, 1, 2, 3].map(() => Math.floor(Math.random() * 0x10000));   // fresh rolls, shared by every cell
	const total = candidates.reduce((s, c) => s + c.prior, 0);
	let best = candidates[0], bestScore = -Infinity;
	for (const ours of candidates) {
		const payoffs = [];
		for (const reply of theirs) {
			const copy = Battle.fromJSON(snapshot);
			copy.restart(() => {});
			copy.log = []; copy.inputLog = []; copy.sentLogPos = 0;   // the copies share the snapshot's arrays too
			copy.resetRNG(seed);
			const [mine, other] = [copy[side], copy[foe.id]];
			mine.clearChoice(); other.clearChoice();
			if (!mine.choose(ours.choice) || !other.choose(reply)) continue;
			try {
				copy.commitChoices();
			} catch (err) {
				continue;
			}
			payoffs.push((healthLeft(mine) - healthLeft(other)) / mine.pokemon.length);
		}
		if (!payoffs.length) continue;
		const mean = payoffs.reduce((a, b) => a + b, 0) / payoffs.length;
		const score = (1 - WORST_WEIGHT) * mean + WORST_WEIGHT * Math.min(...payoffs) + PRIOR_WEIGHT * Math.log(ours.prior / total);
		if (score > bestScore) { bestScore = score; best = ours; }
	}
	return best.choice;
}

module.exports = {search};
