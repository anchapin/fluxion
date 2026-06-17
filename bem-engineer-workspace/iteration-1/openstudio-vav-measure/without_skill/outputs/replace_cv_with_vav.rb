class ReplaceCVWithVAV < OpenStudio::Measure::ModelMeasure
  def name
    return "Replace Constant Volume Terminals with VAV"
  end

  def description
    return "Replaces all AirTerminal:SingleDuct:ConstantVolume:Reheat (and " \
           "no-reheat) objects with AirTerminal:SingleDuct:VAV:Reheat (or " \
           "no-reheat) equivalents. A minimum airflow fraction of 0.3 is " \
           "applied to every new VAV terminal."
  end

  def modeler_description
    return "Iterates over all AirTerminal:SingleDuct:ConstantVolume objects " \
           "in the model, creates a corresponding VAV terminal with the same " \
           "air loop assignment and reheat coil, sets the minimum airflow " \
           "fraction to 0.3, and removes the original constant-volume terminal."
  end

  def arguments(model)
    args = OpenStudio::Measure::OSArgumentVector.new

    min_frac = OpenStudio::Measure::OSArgument.makeDoubleArgument(
      "min_airflow_fraction", true
    )
    min_frac.setDisplayName("Minimum Airflow Fraction")
    min_frac.setDescription(
      "Minimum airflow fraction applied to every new VAV terminal (0.0-1.0)."
    )
    min_frac.setDefaultValue(0.3)
    args << min_frac

    return args
  end

  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)

    unless runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    min_frac = runner.getDoubleArgumentValue("min_airflow_fraction",
                                             user_arguments)
    if min_frac < 0.0 || min_frac > 1.0
      runner.registerError("Minimum airflow fraction must be between 0 and 1.")
      return false
    end

    replaced = 0

    model.getAirTerminalSingleDuctConstantVolumeNoReheats.each do |cv|
      vav = OpenStudio::Model::AirTerminalSingleDuctVAVNoReheat.new(model)
      vav.setName(cv.name.get + " VAV")

      vav.setAvailabilitySchedule(cv.availabilitySchedule)
      vav.setMaximumAirFlowRate(cv.maximumAirFlowRate)
      vav.setZoneMinimumAirFlowFraction(min_frac)

      air_loops = cv.airLoopHVACs
      air_loops.each do |loop|
        demand_nodes = loop.demandComponents
        demand_nodes.each do |node|
          if node.to_Node.is_initialized
            node = node.to_Node.get
            cv_inlet  = cv.inletModelObject
            cv_outlet = cv.outletModelObject
            if cv_inlet && cv_outlet
              inlet_node  = cv_inlet.to_Node.get
              outlet_node = cv_outlet.to_Node.get
              if node == inlet_node || node == outlet_node
                cv.disconnect
                node.disconnect
                next unless node.to_Node.is_initialized
              end
            end
          end
        end

        loop.addBranchForTerminal(vav)
      end

      thermal_zones = cv.thermalZones
      thermal_zones.each do |zone|
        zone.addEquipment(vav)
      end

      runner.registerInfo("Replaced '#{cv.name}' with '#{vav.name}'.")
      cv.remove
      replaced += 1
    end

    model.getAirTerminalSingleDuctConstantVolumeReheats.each do |cv|
      reheat_coil = cv.reheatCoil

      vav = OpenStudio::Model::AirTerminalSingleDuctVAVReheat.new(
        model, model.alwaysOnDiscreteSchedule, reheat_coil
      )
      vav.setName(cv.name.get + " VAV")

      vav.setMaximumAirFlowRate(cv.maximumAirFlowRate)
      vav.setZoneMinimumAirFlowFraction(min_frac)

      if cv.isMaximumReheatAirFlowRateAutosized
        vav.autosizeMaximumReheatAirFlowRate
      else
        vav.setMaximumReheatAirFlowRate(cv.maximumReheatAirFlowRate)
      end

      reheat_s = cv.reheatControlTemperatureSchedule
      if reheat_s.is_initialized
        vav.setReheatControlTemperatureSchedule(reheat_s.get)
      end

      air_loops = cv.airLoopHVACs
      air_loops.each do |loop|
        loop.addBranchForTerminal(vav)
      end

      thermal_zones = cv.thermalZones
      thermal_zones.each do |zone|
        zone.addEquipment(vav)
      end

      runner.registerInfo("Replaced '#{cv.name}' with '#{vav.name}' " \
                          "(reheat coil: #{reheat_coil.name}).")
      cv.remove
      replaced += 1
    end

    if replaced == 0
      runner.registerAsNotApplicable(
        "No constant-volume terminals found in the model."
      )
    else
      runner.registerFinalCondition(
        "Replaced #{replaced} constant-volume terminal(s) with VAV " \
        "terminals (min fraction #{min_frac})."
      )
    end

    return true
  end
end

ReplaceCVWithVAV.new.registerWithApplication
